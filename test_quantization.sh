for w in 1 2 4 8; do
  for a in 1 2 4 8; do
    echo "Running with weight_quant_bits=$w and activation_quant_bits=$a"
    python test.py --weight_quant_bits $w --activation_quant_bits $a
  done
done
