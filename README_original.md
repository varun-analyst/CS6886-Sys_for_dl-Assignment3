# 📦 MobileNet-v2 Quantization Assignment  

---

## 📚 Background  
This repository contains the basic code to perform **Quantization** on the **VGG16** architecture trained on the **CIFAR-10** dataset.  
You can change the **quantization bits** of both the **weights** and **activations**, and evaluate model performance **before** and **after** quantization.  

---

## ▶️ Example Usage  

To perform **8-bit Quantization**, run:  

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
```
📝 Assignment Tasks

✅ Task 1: Train MobileNet-v2

  - Train MobileNet-v2 on the CIFAR-10 / CIFAR-100 dataset.
  
✅ Task 2: Custom Quantization Implementation

  - Write your own custom quantization code to compress the model.

✅ Task 3: Quantization Analysis Report

  - Provide a detailed summary including:
    
    - a. Compression ratio of the model
    
    - b. Compression ratio of the weights
    
    - c. Compression ratio of the activations
    
    - d. Final approximated model size (in MB) after quantization
    
    - e. Upload the complete code to your GitHub repository and share the link on Moodle

⚠️ Important Notes

  - Accuracy matters in quantization.
  
  - 🏆 The submission with the best compression ratio (while maintaining accuracy) will receive the maximum points.
  
  - 📤 Submission Requirements
  
    - Complete implementation of MobileNet-v2 training and quantization
    
    - Detailed analysis report with all required metrics (a–d above)
    
    - Upload the full code to your GitHub repository
    
    - Submit the GitHub repository link on Moodle

📊 Evaluation Criteria

  - Functionality: Working implementation of training + quantization
  
  - Compression Performance: Higher compression ratios score more points
  
  - Accuracy Retention: Maintain performance after quantization
  
  - Code Quality: Clean, well-documented, and organized code
  
  - Analysis: Comprehensive reporting of quantization metrics
