num_samples=1000;
N=100;
d=10;
[num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples);

figure(1)
histogram(num_iters)

figure(2)
histogram(log(bounds_minus_ni))
