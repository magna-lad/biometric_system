import os
import random
import numpy as np
import cv2
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== FREQUENCY DISTRIBUTION FUNCTIONS ==========

def create_frequency_distribution_curve(data, num_classes=20, smooth=True):
    """Creates smooth frequency distribution curve from raw data
    returns x_smooth,y_smooth->300 points in between midpoints to make a smooth curve
        (midpoints of the histograms),frequencies (inside the bins how often the data occurs)"""
    # if there is no data
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return np.array([np.mean(data)]), np.array([1.0]), np.array([np.mean(data)]), np.array([1.0])
    
    bins = np.linspace(np.min(data), np.max(data), num_classes + 1)
    frequencies, bin_edges = np.histogram(data, bins=bins, density=True)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if smooth and len(midpoints) > 3:
        f = interp1d(midpoints, frequencies, kind='cubic', bounds_error=False, fill_value=0)
        x_smooth = np.linspace(midpoints[0], midpoints[-1], 300)
        y_smooth = f(x_smooth)
        return x_smooth, y_smooth, midpoints, frequencies
    else:
        return midpoints, frequencies, midpoints, frequencies

def plot_frequency_distributions(genuine_scores, impostor_scores, title="Score Distribution"):
    """Plot frequency distribution curves for genuine and impostor scores"""
    if len(genuine_scores) > 0:
        x_gen, y_gen, _, _ = create_frequency_distribution_curve(genuine_scores) # creates x_smooth and y_smooth values for a frequency distribuition curve duhh!!
        # Plots the 300 points in between midpoints to make a smooth curve
        plt.plot(x_gen, y_gen, 'b-', linewidth=2, label='Genuine Scores', alpha=0.8)
        # Fills the area under the curve with a light blue color
        plt.fill_between(x_gen, y_gen, alpha=0.3, color='blue')
        # Plots the histogram of genuine scores with 20 bins, alpha for transparency, density=True for normalization
        # and color blue with black edges
        plt.hist(genuine_scores, bins=20, alpha=0.4, density=True, color='blue', 
                edgecolor='black', linewidth=0.5)

        # same for impostor scores
    if len(impostor_scores) > 0:
        x_imp, y_imp, _, _ = create_frequency_distribution_curve(impostor_scores)
        plt.plot(x_imp, y_imp, 'r-', linewidth=2, label='Impostor Scores', alpha=0.8)
        plt.fill_between(x_imp, y_imp, alpha=0.3, color='red')
        plt.hist(impostor_scores, bins=20, alpha=0.4, density=True, color='red', 
                edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Score Value')
    plt.ylabel('Frequency Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def analyze_distribution_characteristics(scores, labels):
    """Analyze statistical characteristics of score distributions"""
    # for two classes: genuine (1) and impostor (0)
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    # if there are no scores for either class, return empty dict
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return {}
    # various statistical measures for both genuine and impostor scores
    analysis = {
        'genuine': {
            'mean': np.mean(genuine_scores),
            'std': np.std(genuine_scores), # standard deviation
            'median': np.median(genuine_scores),
            'skewness': stats.skew(genuine_scores),
            'kurtosis': stats.kurtosis(genuine_scores), # how peaked the distribution is # high kurtosis means more outliers (peaked distribution) leptokurtic distribution
            # low kurtosis means more flat distribution
            'min': np.min(genuine_scores),
            'max': np.max(genuine_scores)
        },
        # similar measures for impostor scores
        'impostor': {
            'mean': np.mean(impostor_scores),
            'std': np.std(impostor_scores),
            'median': np.median(impostor_scores),
            'skewness': stats.skew(impostor_scores),
            'kurtosis': stats.kurtosis(impostor_scores),
            'min': np.min(impostor_scores),
            'max': np.max(impostor_scores)
        }
    }
    
    analysis['separation'] = {
        # how far apart the average of the two distributions are
        'mean_difference': analysis['genuine']['mean'] - analysis['impostor']['mean'],
        # statsical measure of how well the two distributions are separated
        'decidability_index': calculate_decidability_index(genuine_scores, impostor_scores)
    }
    return analysis

# ========== QUALITY ASSESSMENT ==========

# computing image quality using laplacian (as brisque was not available in the environment), i did on kaggle kernel

# how it works:
def ompute_image_quality(img):
    """Compute Image quality score"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img # if the image is in 3 chanels. convert it to grayscale
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var() # compute the laplacian variance of the image by performing a laplacian filter on the image,that is a second order derivative filter, and then computing the variance of the result
    # if the variance is too low, the image is blurry or has low quality
    beta = np.clip(100.0 - lap_var, 0.0, 100.0)
    return beta

# but the original code uses brisque quality score, so we will use that as well
# can also be done by using brisque library from github but doesnt give better results than this    
def compute_brisque_quality(img):
    """
    Custom BRISQUE implementation that works without external dependencies
    Based on the paper's methodology
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    gray = gray.astype(np.float64)
    
    # Step 1: Compute local mean and variance
    mu = cv2.GaussianBlur(gray, (7, 7), 1.166) # mean filter with a kernel size of 7x7 and sigma of 1.166
    # mu_sq is the mean of the square of the image
    # sigma is the standard deviation of the image
    mu_sq = cv2.GaussianBlur(gray**2, (7, 7), 1.166)
    sigma = np.sqrt(np.abs(mu_sq - mu**2))
    
    # Step 2: Compute MSCN (Mean Subtracted Contrast Normalized) coefficients
    mscn = (gray - mu) / (sigma + 1)
    
    # Step 3: Extract features from MSCN coefficients
    features = []
    
    # Compute moments of MSCN distribution
    features.append(np.mean(mscn))
    features.append(np.var(mscn))
    features.append(np.mean(np.abs(mscn)))
    features.append(np.mean(mscn**2))
    
    # Compute features from horizontal, vertical, and diagonal neighbors
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for shift in shifts:
        shifted_mscn = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
        pairwise_products = mscn * shifted_mscn
        
        features.append(np.mean(pairwise_products))
        features.append(np.var(pairwise_products))
        features.append(np.mean(np.abs(pairwise_products)))
    
    # Step 4: Compute quality score (simplified mapping)
    # This is a simplified version - the full BRISQUE uses SVR
    feature_vector = np.array(features)
    
    # Normalize features
    feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
    
    # Simple quality mapping (0-100 scale)
    quality_score = np.clip(50 + 25 * np.tanh(np.mean(feature_vector)), 0, 100)
    
    return quality_score

def compute_reliability_factor(beta):
    """Compute reliability factor from quality score"""
    return 1.0 - (beta / 100.0)

# ========== FEATURE EXTRACTION ==========

def extract_gabor_features(img, scales=5, orientations=8):
    """Extract Gabor features for face recognition"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    features = []
    
    for scale in range(scales):
        for orientation in range(orientations):
            theta = orientation * np.pi / orientations
            sigma = 2 ** scale
            frequency = 0.05 + scale * 0.05
            
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = cv2.getGaborKernel((kernel_size, kernel_size), 
                                     sigma, theta, 2*np.pi*frequency, 0.5)
            
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            features.extend([filtered.mean(), filtered.std()])
    
    return np.array(features)

def match_score_face(img1, img2):
    """Compute face matching score using Gabor features"""
    feat1 = extract_gabor_features(img1)
    feat2 = extract_gabor_features(img2)
    
    if len(feat1) == 0 or len(feat2) == 0:
        return 0.0
    
    feat1 = (feat1 - feat1.mean()) / (feat1.std() + 1e-8)
    feat2 = (feat2 - feat2.mean()) / (feat2.std() + 1e-8)
    
    correlation = np.corrcoef(feat1, feat2)[0, 1]
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0

def match_score_fingerprint(img1, img2):
    """Compute fingerprint matching score using SIFT features"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 0.0
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    score = len(good_matches) / max(len(kp1), len(kp2), 1)
    return min(1.0, score)

def match_score_iris(img1, img2):
    """Compute iris matching score using Hamming distance + RBF"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    _, bin1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    min_h, min_w = min(bin1.shape[0], bin2.shape[0]), min(bin1.shape[1], bin2.shape[1])
    bin1 = bin1[:min_h, :min_w]
    bin2 = bin2[:min_h, :min_w]
    
    xor = cv2.bitwise_xor(bin1, bin2)
    hamming_dist = np.sum(xor > 0) / (min_h * min_w)
    
    sigma = 0.2
    score = np.exp(-(hamming_dist**2) / (2 * sigma**2))
    return score

# ========== ADAPTIVE FUSION ==========

def adaptive_score_fusion(scores, betas, tau):
    """Adaptive score fusion algorithm from the paper"""
    scores = np.array(scores)
    betas = np.array(betas)
    
    # Reliability factors
    alphas = np.array([compute_reliability_factor(b) for b in betas])
    
    # Weighted scores
    omegas = alphas * scores
    
    # Adaptive scores
    phis = omegas - np.sqrt(np.maximum(0, tau**2 - omegas**2))
    
    # Confidence factors
    lambdas = np.abs(phis - tau)
    
    # Optimization factor
    xi = np.sum(lambdas * phis)
    
    # Final fused score
    fused_score = np.mean(phis) + xi / len(scores)
    
    return fused_score, alphas, omegas, phis, lambdas, xi

def calculate_decidability_index(genuine, impostor):
    """Calculate decidability index"""
    if len(genuine) == 0 or len(impostor) == 0:
        return 0.0
    
    mu_g, mu_i = np.mean(genuine), np.mean(impostor)
    std_g, std_i = np.std(genuine), np.std(impostor)
    
    if (std_g**2 + std_i**2) == 0:
        return 0.0
    
    return abs(mu_g - mu_i) / np.sqrt(0.5 * (std_g**2 + std_i**2))

# ========== DATA LOADING ==========

def load_users(data_dir):
    """Load user data from directory structure"""
    users = {}
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return users
    
    for uid in os.listdir(data_dir):
        path = os.path.join(data_dir, uid)
        if not os.path.isdir(path):
            continue
        
        def load_modality(sub, size):
            imgs = []
            d = os.path.join(path, sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        img_path = os.path.join(d, f)
                        img = cv2.imread(img_path)
                        if img is not None:
                            imgs.append(cv2.resize(img, size))
            return imgs
        
        users[uid] = {
            'face': load_modality('face', (128, 128)),
            'finger': load_modality('Fingerprint', (96, 96)),
            'iris_left': load_modality('left', (64, 64)),
            'iris_right': load_modality('right', (64, 64))
        }
    
    return users

# ========== MAIN IMPLEMENTATION ==========

def main():
    # Update this path to your dataset location
    data_dir = "/kaggle/input/pehla-dataset/iris_fingerprint_face dataset"
    
    users = load_users(data_dir)
    uids = list(users.keys())
    
    if len(uids) < 2:
        print("Need at least 2 users for evaluation")
        return
    
    print(f"Loaded {len(uids)} users")
    
    all_scores, all_betas, all_labels = [], [], []
    
    # Generate comparison pairs
    for probe_id in tqdm(uids, desc='Processing users'):
        pr = users[probe_id]
        
        if not (pr['face'] and pr['finger'] and (pr['iris_left'] or pr['iris_right'])):
            continue
        
        for gal_id in uids:
            ga = users[gal_id]
            
            if not (ga['face'] and ga['finger'] and (ga['iris_left'] or ga['iris_right'])):
                continue
            
            try:
                # Select random images
                fp = random.choice(pr['face'])
                fg = random.choice(ga['face'])
                pp = random.choice(pr['finger'])
                pg = random.choice(ga['finger'])
                ip = random.choice(pr['iris_left'] or pr['iris_right'])
                ig = random.choice(ga['iris_left'] or ga['iris_right'])
                
                # Compute match scores
                sf = match_score_face(fp, fg)
                sp = match_score_fingerprint(pp, pg)
                si = match_score_iris(ip, ig)
                
                # Compute quality scores
                bf = compute_brisque_quality(fg)
                bp = compute_brisque_quality(pg)
                bi = compute_brisque_quality(ig)
                
                all_scores.append([sf, sp, si])
                all_betas.append([bf, bp, bi])
                all_labels.append(1 if probe_id == gal_id else 0)
                
            except Exception as e:
                print(f"Error processing {probe_id}-{gal_id}: {e}")
                continue
    
    if len(all_scores) == 0:
        print("No valid scores generated")
        return
    
    scores = np.array(all_scores)
    betas = np.array(all_betas)
    labels = np.array(all_labels)
    
    print(f"Generated {len(scores)} score pairs")
    print(f"Genuine pairs: {np.sum(labels)}, Impostor pairs: {np.sum(1-labels)}")
    
    # Optimize threshold
    genuine_scores = scores[labels == 1]
    tau = np.mean(genuine_scores) if len(genuine_scores) > 0 else 0.5
    print(f"Optimal threshold τ = {tau:.4f}")
    
    # Apply adaptive fusion
    fused_scores = []
    for i in range(len(scores)):
        fused, _, _, _, _, _ = adaptive_score_fusion(scores[i], betas[i], tau)
        fused_scores.append(fused)
    
    fused_scores = np.array(fused_scores)
    
    # Normalize scores
    if fused_scores.max() > fused_scores.min():
        fused_scores = (fused_scores - fused_scores.min()) / (fused_scores.max() - fused_scores.min())
    
    # Evaluation
    fpr, tpr, thresholds = roc_curve(labels, fused_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find EER
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # Accuracy
    predictions = (fused_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    
    # Decidability Index
    genuine_fused = fused_scores[labels == 1]
    impostor_fused = fused_scores[labels == 0]
    di = calculate_decidability_index(genuine_fused, impostor_fused)
    
    # Results
    print(f"\n=== RESULTS ===")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Decidability Index: {di:.4f}")
    
    # Distribution analysis
    analysis = analyze_distribution_characteristics(fused_scores, labels)
    if analysis:
        print(f"\n=== DISTRIBUTION ANALYSIS ===")
        print(f"Genuine - Mean: {analysis['genuine']['mean']:.4f}, Std: {analysis['genuine']['std']:.4f}")
        print(f"Impostor - Mean: {analysis['impostor']['mean']:.4f}, Std: {analysis['impostor']['std']:.4f}")
        print(f"Separation - Mean Diff: {analysis['separation']['mean_difference']:.4f}")
    
    # ========== FIXED VISUALIZATION SECTION ==========
    
    # Main performance plots
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    axes[0,0].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    axes[0,0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0,0].scatter(fpr[eer_idx], tpr[eer_idx], color='red', s=100, label=f'EER={eer:.3f}')
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Score distributions
    if len(genuine_fused) > 0:
        axes[0,1].hist(genuine_fused, bins=30, alpha=0.7, label='Genuine', density=True, color='blue')
    if len(impostor_fused) > 0:
        axes[0,1].hist(impostor_fused, bins=30, alpha=0.7, label='Impostor', density=True, color='red')
    axes[0,1].axvline(eer_threshold, color='green', linestyle='--', label=f'Threshold={eer_threshold:.3f}')
    axes[0,1].set_xlabel('Fused Score')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Score Distributions')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Individual modality performance
    modalities = ['Face', 'Finger', 'Iris']
    for i, modality in enumerate(modalities):
        if len(scores[:, i]) > 0:
            mod_fpr, mod_tpr, _ = roc_curve(labels, scores[:, i])
            mod_auc = auc(mod_fpr, mod_tpr)
            axes[1,0].plot(mod_fpr, mod_tpr, label=f'{modality} (AUC={mod_auc:.3f})')
    
    axes[1,0].plot(fpr, tpr, 'k-', linewidth=2, label=f'Fused (AUC={roc_auc:.3f})')
    axes[1,0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('Individual vs Fused Performance')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Fused frequency distribution
    plot_frequency_distributions(genuine_fused, impostor_fused, "Fused Score Distribution")
    axes[1,1] = plt.gca()
    
    plt.tight_layout()
    plt.show()
    
    # Individual modality frequency distributions
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, modality in enumerate(modalities):
        plt.sca(axes2[i])
        genuine_mod = scores[labels == 1, i]
        impostor_mod = scores[labels == 0, i]
        plot_frequency_distributions(genuine_mod, impostor_mod, f"{modality} Score Distribution")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
