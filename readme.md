# Training MobileNet-v2 on CIFAR-10 and applying model compression techniques

1. Environment setup: Setup `pip` environment and install required packages with required packages

```bash
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2. Run the `train_MobileNet.py` code.
```bash
python train_MobileNet.py
```

3. Run `test_quantization.sh`
```bash
bash test_quantization.sh`
```

4. Manually edit `data` in  `wandb_plot` and run
```bash
python wandb_plot.py
```
Observe results in `wandb_plot.png`

5. Pick final model and analyse.
Ex:
```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
```