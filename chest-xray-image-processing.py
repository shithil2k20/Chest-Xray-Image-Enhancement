# ============================================================ 
# COMPLETE BIOMEDICAL IMAGE PROCESSING PROJECT 
# ============================================================ 
# -------------------- IMPORT LIBRARIES -------------------- 
import os 
import glob 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pywt 
from scipy.ndimage import maximum_filter, minimum_filter 
from scipy.linalg import hadamard 
from skimage import util 
from skimage.metrics import ( 
mean_squared_error, 
peak_signal_noise_ratio, 
structural_similarity 
) 
from matplotlib.backends.backend_pdf import PdfPages 
plt.rcParams['figure.figsize'] = (8, 8) 
plt.rcParams['image.cmap'] = 'gray' 
# -------------------- DATASET PATH -------------------- 
DATASET_PATH = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia" 
image_folder = os.path.join( 
DATASET_PATH, 
'chest_xray', 
'test', 
'NORMAL' 
) 
image_files = glob.glob(os.path.join(image_folder, '*.jpeg')) 
if len(image_files) == 0: 
raise FileNotFoundError("No images found. Check the dataset path.") 
print(f"Number of images found: {len(image_files)}") 
print("Using image:", image_files[0]) 
# -------------------- LOAD ORIGINAL IMAGE -------------------- 
image_path = image_files[0] 
original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
original = cv2.resize(original, (512, 512)) 
# ============================================================ 
# IMAGE INTERPOLATION 
# ============================================================ 
small = cv2.resize(original, (256, 256)) 
bilinear_img = cv2.resize(small, (512, 512), interpolation=cv2.INTER_LINEAR) 
bicubic_img = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC) 
# ============================================================ 
# INTENSITY TRANSFORMATIONS 
# ============================================================ 
negative_img = 255 - original 
contrast_stretched = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX) 
intensity_sliced = np.where((original >= 100) & (original <= 180), 255, 
0).astype(np.uint8) 
# ============================================================ 
# NOISE ADDITION 
# ============================================================ 
np.random.seed(42) 
noisy = util.random_noise(original, mode='gaussian', var=0.01) 
noisy = (noisy * 255).astype(np.uint8) 
# ============================================================ 
# FILTERING 
# ============================================================ 
mean_filtered = cv2.blur(noisy, (3, 3)) 
median_filtered = cv2.medianBlur(noisy, 3) 
gaussian_filtered = cv2.GaussianBlur(noisy, (3, 3), 0) 
# Geometric mean filter 
kernel_size = 3 
pad = kernel_size // 2 
padded = np.pad(noisy.astype(np.float64), pad, mode='reflect') 
geometric_mean_filtered = np.zeros_like(noisy, dtype=np.float64) 
for i in range(noisy.shape[0]): 
for j in range(noisy.shape[1]): 
region = padded[i:i+kernel_size, j:j+kernel_size] 
geometric_mean_filtered[i, j] = np.exp(np.mean(np.log(region + 1))) - 1 
geometric_mean_filtered = np.clip(geometric_mean_filtered, 0, 255).astype(np.uint8) 
max_filtered = maximum_filter(noisy, size=3) 
min_filtered = minimum_filter(noisy, size=3) 
# ============================================================ 
# QUANTITATIVE METRICS 
# ============================================================ 
def calculate_snr(reference, processed): 
signal_power = np.mean(reference.astype(np.float64) ** 2) 
noise_power = np.mean((reference.astype(np.float64) - processed.astype(np.float64)) 
** 2) 
if noise_power == 0: 
return np.inf 
return 10 * np.log10(signal_power / noise_power) 
def calculate_cnr(image, mask): 
foreground = image[mask > 0] 
background = image[mask == 0] 
if len(foreground) == 0 or len(background) == 0: 
return np.nan 
sigma = np.std(background) 
if sigma == 0: 
return np.nan 
return abs(np.mean(foreground) - np.mean(background)) / sigma 
def evaluate(reference, processed): 
return { 
'MSE': mean_squared_error(reference, processed), 
'PSNR (dB)': peak_signal_noise_ratio(reference, processed, data_range=255), 
'SSIM': structural_similarity(reference, processed, data_range=255), 
'SNR (dB)': calculate_snr(reference, processed) 
} 
# Filter comparison 
results = { 
'Mean Filter': evaluate(original, mean_filtered), 
'Median Filter': evaluate(original, median_filtered), 
'Gaussian Filter': evaluate(original, gaussian_filtered), 
'Geometric Mean': evaluate(original, geometric_mean_filtered), 
'Max Filter': evaluate(original, max_filtered), 
'Min Filter': evaluate(original, min_filtered) 
} 
results_df = pd.DataFrame(results).T 
print("\nFilter Comparison:") 
print(results_df) 
# Best filtered image 
best_filtered = gaussian_filtered 
# ============================================================ 
# HISTOGRAM EQUALIZATION 
# ============================================================ 
enhanced = cv2.equalizeHist(best_filtered) 
# ============================================================ 
# EDGE DETECTION 
# ============================================================ 
sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3) 
sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3) 
sobel = np.uint8(np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 255)) 
# Prewitt 
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) 
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) 
prewitt_x = cv2.filter2D(enhanced, -1, kernelx) 
prewitt_y = cv2.filter2D(enhanced, -1, kernely) 
prewitt = np.uint8(np.clip(np.sqrt(prewitt_x.astype(np.float64)**2 + 
prewitt_y.astype(np.float64)**2), 0, 255)) 
# ============================================================ 
# SEGMENTATION 
# ============================================================ 
_, segmentation = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + 
cv2.THRESH_OTSU) 
kernel = np.ones((5, 5), np.uint8) 
segmentation_clean = cv2.morphologyEx(segmentation, cv2.MORPH_OPEN, kernel) 
segmentation_clean = cv2.morphologyEx(segmentation_clean, cv2.MORPH_CLOSE, 
kernel) 
# ============================================================ 
# TRANSFORMS 
# ============================================================ 
# DCT 
img_float = np.float32(enhanced) / 255.0 
dct = cv2.dct(img_float) 
dct_image = cv2.normalize(np.log(np.abs(dct) + 1), None, 0, 255, 
cv2.NORM_MINMAX).astype(np.uint8) 
# FFT 
f = np.fft.fft2(enhanced) 
fshift = np.fft.fftshift(f) 
fft_image = cv2.normalize(20 * np.log(np.abs(fshift) + 1), None, 0, 255, 
cv2.NORM_MINMAX).astype(np.uint8) 
# Haar 
coeffs2 = pywt.dwt2(enhanced, 'haar') 
LL, (LH, HL, HH) = coeffs2 
ll_image = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
# Hadamard (use 256x256) 
hadamard_input = cv2.resize(enhanced, (256, 256)).astype(np.float64) 
H = hadamard(256) 
hadamard_transformed = H @ hadamard_input @ H 
hadamard_image = cv2.normalize( 
np.log(np.abs(hadamard_transformed) + 1), 
None, 0, 255, cv2.NORM_MINMAX 
).astype(np.uint8) 
# ============================================================ 
# METRICS 
# ============================================================ 
cnr_value = calculate_cnr(enhanced, segmentation_clean) 
print(f"\nCNR: {cnr_value:.4f}") 
enhancement_metrics = pd.DataFrame( 
[evaluate(original, enhanced)], 
index=['Enhanced Image'] 
) 
print("\nEnhancement Metrics:") 
print(enhancement_metrics) 
# ============================================================ 
# SAVE OUTPUTS 
# ============================================================ 
os.makedirs('outputs', exist_ok=True) 
save_dict = { 
'original.png': original, 
'bilinear.png': bilinear_img, 
'bicubic.png': bicubic_img, 
'negative.png': negative_img, 
'contrast_stretching.png': contrast_stretched, 
'intensity_slicing.png': intensity_sliced, 
'histogram_equalization.png': enhanced, 
'noisy.png': noisy, 
'mean_filtered.png': mean_filtered, 
'median_filtered.png': median_filtered, 
'gaussian_filtered.png': gaussian_filtered, 
'geometric_mean.png': geometric_mean_filtered, 
'max_filter.png': max_filtered, 
'min_filter.png': min_filtered, 
'sobel_edges.png': sobel, 
'prewitt_edges.png': prewitt, 
'threshold_segmentation.png': segmentation_clean, 
'dct.png': dct_image, 
'fft_spectrum.png': fft_image, 
'haar_LL.png': ll_image, 
'hadamard.png': hadamard_image 
} 
for filename, image in save_dict.items(): 
cv2.imwrite(os.path.join('outputs', filename), image) 
results_df.to_csv('outputs/filter_comparison.csv') 
enhancement_metrics.to_csv('outputs/enhancement_metrics.csv') 
with open('outputs/cnr_value.txt', 'w') as f: 
f.write(f'CNR = {cnr_value:.4f}') 
# ============================================================ 
# CREATE PDF FILES (4 IMAGES PER PAGE) 
# ============================================================ 
print_images = list(save_dict.values()) 
print_titles = [ 
os.path.splitext(name)[0].replace('_', ' ').title() 
for name in save_dict.keys() 
] 
images_per_pdf = 4 
num_pdfs = int(np.ceil(len(print_images) / images_per_pdf)) 
for pdf_index in range(num_pdfs): 
start = pdf_index * images_per_pdf 
end = start + images_per_pdf 
current_images = print_images[start:end] 
current_titles = print_titles[start:end] 
pdf_filename = f'outputs/page_{pdf_index + 1}.pdf' 
with PdfPages(pdf_filename) as pdf: 
fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69)) 
axes = axes.ravel() 
for i in range(4): 
if i < len(current_images): 
axes[i].imshow(current_images[i], cmap='gray') 
axes[i].set_title(current_titles[i], fontsize=12) 
axes[i].axis('off') 
else: 
axes[i].axis('off') 
plt.tight_layout() 
pdf.savefig(fig, bbox_inches='tight') 
plt.close(fig) 
# ============================================================ 
# FINAL SUMMARY 
# ============================================================ 
print(f"\nCreated {num_pdfs} PDF files.") 
print("Project completed successfully!") 
print("All outputs are saved in the outputs/ folder.") 