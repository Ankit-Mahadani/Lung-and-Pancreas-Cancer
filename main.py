"""
train_binary_unified_fixed.py
Robust, end-to-end binary pipeline for Cancer vs Normal with improved auto-detection of dataset location

Key fixes made now:
 - Auto-detect dataset folder when BASE_DIR points to wrong place: searches for 'dataset' folders under the project tree and selects the one containing images.
 - Clear diagnostic printout listing candidate dataset folders if none found.
 - More robust loader with both legacy and flattened layouts supported.
 - Safe feature extraction and training flow unchanged.

Run from project root. If you still get "No images found", the script will print candidate folders to inspect.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, accuracy_score
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# ML libs
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
# default (relative) - we'll auto-detect if this doesn't work
BASE_DIR = os.path.abspath("dataset")
OUTPUT_DIR = "models_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_STATE = 42
XGB_SAMPLE_LIMIT = None
EPOCHS_CNN = 8
EPOCHS_RESNET_FROZEN = 5
EPOCHS_RESNET_FINETUNE = 5

# non-blocking plotting for headless
plt.switch_backend('agg')

# ---------------- Utilities ----------------

def is_image_file(fname):
    return fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

def find_candidate_dataset_dirs(start_dir=None, max_depth=3):
    """
    Search upwards and nearby for folders named 'dataset' or folders that look like expected dataset layout.
    Returns list of candidate absolute paths.
    """
    candidates = set()
    if start_dir is None:
        start_dir = os.getcwd()

    # look upwards
    p = start_dir
    for _ in range(max_depth):
        cand = os.path.join(p, 'dataset')
        if os.path.isdir(cand):
            candidates.add(os.path.abspath(cand))
        p = os.path.dirname(p)
        if p == '' or p == os.path.dirname(p):
            break

    # look recursively for 'dataset' in current tree (cheap, shallow)
    for fp in glob(os.path.join(start_dir, '**', 'dataset'), recursive=True):
        if os.path.isdir(fp):
            candidates.add(os.path.abspath(fp))

    # finally, include sibling folder names that match common project names
    likely_names = ['LUNG AND PANCREAS CANCER', 'LUNG_AND_PANCREAS_CANCER', 'lung and pancreas cancer']
    base = os.path.dirname(start_dir)
    for name in likely_names:
        cand = os.path.join(base, name, 'dataset')
        if os.path.isdir(cand):
            candidates.add(os.path.abspath(cand))

    return sorted(candidates)

# ---------------- Data loader (supports two layouts + auto-detect) ----------------

def load_paths_and_labels(base_dir=None, verbose=True):
    """
    Flexible loader. Tries provided base_dir, otherwise attempts auto-detection.
    Returns (paths, labels) where 0=Normal, 1=Cancer
    """
    tried = []
    if base_dir is None:
        base_dir = BASE_DIR

    # If the provided base_dir doesn't contain images, try auto-detection
    def try_load(bdir):
        tried.append(bdir)
        if not os.path.isdir(bdir):
            if verbose:
                print(f"[DEBUG] Not a directory: {bdir}")
            return None, None

        paths = []
        labels = []
        details = {}

        # Option B: flattened layout
        cancer_root = os.path.join(bdir, 'Cancer')
        normal_root = os.path.join(bdir, 'Normal')

        if os.path.isdir(cancer_root) and os.path.isdir(normal_root):
            for fp in glob(os.path.join(cancer_root, '**'), recursive=True):
                if os.path.isfile(fp) and is_image_file(fp):
                    paths.append(fp); labels.append(1)
                    details.setdefault('Cancer', []).append(fp)
            for fp in glob(os.path.join(normal_root, '**'), recursive=True):
                if os.path.isfile(fp) and is_image_file(fp):
                    paths.append(fp); labels.append(0)
                    details.setdefault('Normal', []).append(fp)
            if verbose:
                print(f"[DEBUG] Detected flattened layout at: {bdir}")
        else:
            # Option A: legacy layout
            cancer_dirs = [
                os.path.join(bdir, 'Lung', 'malignant'),
                os.path.join(bdir, 'Lung', 'benign'),
                os.path.join(bdir, 'Pancreas', 'cancer')
            ]
            normal_dirs = [
                os.path.join(bdir, 'Lung', 'normal'),
                os.path.join(bdir, 'Normal', 'normal'),
                os.path.join(bdir, 'Normal', 'Pancreas_normal')
            ]
            for d in cancer_dirs:
                if os.path.isdir(d):
                    found = [f for f in glob(os.path.join(d, '**'), recursive=True) if os.path.isfile(f) and is_image_file(f)]
                    for f in found:
                        paths.append(f); labels.append(1)
                    details[os.path.relpath(d, bdir)] = found
                elif verbose:
                    print(f"[DEBUG] Missing cancer folder (ok if not used): {d}")
            for d in normal_dirs:
                if os.path.isdir(d):
                    found = [f for f in glob(os.path.join(d, '**'), recursive=True) if os.path.isfile(f) and is_image_file(f)]
                    for f in found:
                        paths.append(f); labels.append(0)
                    details[os.path.relpath(d, bdir)] = found
                elif verbose:
                    print(f"[DEBUG] Missing normal folder (ok if not used): {d}")

        if len(paths) == 0:
            # recursive fallback: any image under bdir, infer label by keyword
            for fp in glob(os.path.join(bdir, '**'), recursive=True):
                if os.path.isfile(fp) and is_image_file(fp):
                    low = fp.lower()
                    lbl = 1 if any(k in low for k in ('malignant', 'benign', 'cancer')) else 0
                    paths.append(fp); labels.append(lbl)
                    details.setdefault('recursive', []).append(fp)

        if len(paths) == 0:
            return None, None

        paths = np.array(paths)
        labels = np.array(labels, dtype=np.int32)

        if verbose:
            print(f"[DEBUG] Loaded from: {bdir}")
            print(f"[DEBUG] Total images found: {len(paths)} | Normal: {int(np.sum(labels==0))} | Cancer: {int(np.sum(labels==1))}")
            if len(details) > 0:
                print("[DEBUG] Example folders and counts:")
                for k, v in list(details.items())[:10]:
                    print(f"  {k}: {len(v)}")

        return paths, labels

    # try the given base_dir first
    res = try_load(base_dir)
    if res[0] is not None:
        return res

    # attempt to auto-detect dataset folders nearby
    candidates = find_candidate_dataset_dirs()
    if verbose:
        print(f"[DEBUG] Could not load from provided BASE_DIR: {base_dir}")
        if candidates:
            print(f"[DEBUG] Found candidate dataset directories to try: {len(candidates)}")
            for c in candidates:
                print("  ", c)
        else:
            print("[DEBUG] No candidate 'dataset' folders found in nearby paths.")

    for cand in candidates:
        res = try_load(cand)
        if res[0] is not None:
            return res

    # last resort: try a heuristic global search for folders that include 'Lung' or 'Pancreas'
    if verbose:
        print('[DEBUG] Performing broader scan for directories containing "Lung" or "Pancreas"...')
    root = os.getcwd()
    broader = []
    for fp in glob(os.path.join(root, '**'), recursive=True):
        if os.path.isdir(fp) and any(x in os.path.basename(fp).lower() for x in ('lung', 'pancreas')):
            broader.append(fp)
    if verbose:
        print(f"[DEBUG] Broader candidates found: {len(broader)}")
    for b in broader:
        res = try_load(b)
        if res[0] is not None:
            return res

    # failed
    print('[ERROR] Tried the following candidate base dirs:')
    for t in tried:
        print('  ', t)
    raise ValueError(f"No images found. Check base_dir and folder names. Tried candidates listed above.")

# ---------------- Preprocessing ----------------

def preprocess_opencv_rgb(path, target_size=IMAGE_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unable to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype('float32') / 255.0

# ---------------- XGBoost feature extraction ----------------

def extract_features_for_xg(path):
    try:
        img_rgb = preprocess_opencv_rgb(path)
    except Exception as e:
        print(f"[WARN] Skipping unreadable image: {path} -> {e}")
        return None
    img_uint8 = (img_rgb * 255).astype('uint8')
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    hog_feat = hog(img_gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)

    chans = cv2.split(img_uint8)
    hist_feats = []
    for ch in chans:
        h = cv2.calcHist([ch], [0], None, [32], [0,256]).flatten()
        if h.sum() > 0:
            h = h / h.sum()
        hist_feats.append(h)
    hist_feats = np.concatenate(hist_feats)

    mean_v = float(img_gray.mean())
    std_v = float(img_gray.std())
    feat = np.concatenate([hog_feat, hist_feats, [mean_v, std_v]])
    return feat


def build_feature_matrix(paths, sample_limit=None):
    if sample_limit:
        paths = paths[:sample_limit]
    feats = []
    good_paths = []
    for p in tqdm(paths, desc='Extracting features'):
        f = extract_features_for_xg(p)
        if f is None:
            continue
        feats.append(f)
        good_paths.append(p)
    if len(feats) == 0:
        raise ValueError('No valid features extracted. Check images.')
    feats = np.vstack(feats)
    return feats, np.array(good_paths)

# ---------------- Small CNN builder ----------------

def build_small_cnn(input_shape=(224,224,3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------- Main workflow ----------------

def main():
    print('Loading dataset paths...')
    paths, labels = load_paths_and_labels(BASE_DIR)

    # check that both classes exist
    if len(np.unique(labels)) < 2:
        raise SystemExit('ERROR: Need at least two classes (Normal and Cancer).')

    # split for XGBoost feature extraction
    p_train, p_test, y_train, y_test = train_test_split(paths, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)

    print('\n-> Extracting features for XGBoost...')
    X_train_feats, good_train_paths = build_feature_matrix(p_train, sample_limit=XGB_SAMPLE_LIMIT)
    X_test_feats, good_test_paths = build_feature_matrix(p_test, sample_limit=XGB_SAMPLE_LIMIT)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_feats)
    X_test_s = scaler.transform(X_test_feats)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'xgb_scaler.joblib'))

    print('Fitting XGBoost...')
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, random_state=RANDOM_STATE)
    xgb_clf.fit(X_train_s, y_train[:len(X_train_s)])
    joblib.dump(xgb_clf, os.path.join(OUTPUT_DIR, 'xgb_model.joblib'))

    preds_xgb = xgb_clf.predict(X_test_s)
    print('XGBoost Accuracy:', accuracy_score(y_test[:len(preds_xgb)], preds_xgb))
    print(classification_report(y_test[:len(preds_xgb)], preds_xgb, target_names=['Normal','Cancer']))
    try:
        ConfusionMatrixDisplay(confusion_matrix(y_test[:len(preds_xgb)], preds_xgb), display_labels=['Normal','Cancer']).plot()
        plt.title('XGBoost Confusion Matrix')
        plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_confusion.png'))
        plt.clf()
    except Exception:
        pass

    # ---------- Prepare folders for ImageDataGenerator ----------
    print('\n-> Preparing folders for ImageDataGenerator...')
    train_dir = 'train_dir'
    val_dir = 'val_dir'
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    per_class_paths = {0: [], 1: []}
    for p, lab in zip(paths, labels):
        per_class_paths[lab].append(p)

    for lab, plist in per_class_paths.items():
        if len(plist) == 0:
            continue
        tlist, vlist = train_test_split(plist, test_size=0.2, random_state=RANDOM_STATE, stratify=None)
        label_name = 'Normal' if lab == 0 else 'Cancer'
        os.makedirs(os.path.join(train_dir, label_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label_name), exist_ok=True)
        for p in tlist:
            dst = os.path.join(train_dir, label_name, os.path.basename(p))
            shutil.copy2(p, dst)
        for p in vlist:
            dst = os.path.join(val_dir, label_name, os.path.basename(p))
            shutil.copy2(p, dst)

    for lab in ['Normal','Cancer']:
        tr_count = sum(len(files) for _, _, files in os.walk(os.path.join(train_dir, lab))) if os.path.isdir(os.path.join(train_dir, lab)) else 0
        val_count = sum(len(files) for _, _, files in os.walk(os.path.join(val_dir, lab))) if os.path.isdir(os.path.join(val_dir, lab)) else 0
        print(f'Train/{lab}: {tr_count}, Val/{lab}: {val_count}')

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

    # ---------- Train small CNN ----------
    print('\n-> Training small CNN...')
    num_classes = train_gen.num_classes
    cnn = build_small_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=num_classes)
    cnn.summary()
    cnn.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_CNN)
    cnn.save(os.path.join(OUTPUT_DIR, 'cnn_binary.h5'))

    val_gen.reset()
    preds = cnn.predict(val_gen, verbose=1)
    y_true = val_gen.classes
    y_pred = np.argmax(preds, axis=1)
    print('CNN Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))
    try:
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=list(train_gen.class_indices.keys())).plot()
        plt.title('CNN Confusion Matrix')
        plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_confusion.png'))
        plt.clf()
    except Exception:
        pass

    # ---------- ResNet50 transfer learning ----------
    print('\n-> Training ResNet50 (transfer learning)...')
    try:
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    except Exception as e:
        print(f"[WARN] Could not load ResNet50 imagenet weights ({e}), falling back to random init.")
        base = ResNet50(weights=None, include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    for layer in base.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model_resnet = Model(inputs=base.input, outputs=preds)
    model_resnet.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model_resnet.summary()

    model_resnet.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_RESNET_FROZEN)

    for layer in base.layers[-20:]:
        layer.trainable = True
    model_resnet.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model_resnet.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_RESNET_FINETUNE)
    model_resnet.save(os.path.join(OUTPUT_DIR, 'resnet_binary.h5'))

    val_gen.reset()
    preds_res = model_resnet.predict(val_gen, verbose=1)
    y_pred_res = np.argmax(preds_res, axis=1)
    print('ResNet Accuracy:', accuracy_score(y_true, y_pred_res))
    print(classification_report(y_true, y_pred_res, target_names=list(train_gen.class_indices.keys())))
    try:
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred_res), display_labels=list(train_gen.class_indices.keys())).plot()
        plt.title('ResNet Confusion Matrix')
        plt.savefig(os.path.join(OUTPUT_DIR, 'resnet_confusion.png'))
        plt.clf()
    except Exception:
        pass

    joblib.dump({'class_indices': train_gen.class_indices}, os.path.join(OUTPUT_DIR, 'class_indices.joblib'))
    joblib.dump({'scaler': scaler}, os.path.join(OUTPUT_DIR, 'xgb_scaler.joblib'))
    print('\n✅ All artifacts saved under:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
command > output.txt