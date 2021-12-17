# Benckmark model
for d in ./shaked_imgs/*/ ; do
    python3 figure_ground_nogpu.py "$d"
done
