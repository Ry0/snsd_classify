reset
set terminal png
set output "dropout.png"
set style data lines
set key right

# Test accuracy vs. training time
set yrange [0:1]
set ytics 0.1
set ylabel "Accuracy"
set xlabel "Iterations"
plot "dropout.log.test" using 1:3 with lines lt 1 lc rgb "#000000" lw 1.5 title "test accuracy",\
"dropout.log.train" using 1:4 with lines lt 1 lc rgb "#FF2D4C" lw 1.5  title "train accuracy"