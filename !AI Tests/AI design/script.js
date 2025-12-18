const net = new brain.NeuralNetworkGPU({
//   hiddenLayers: [3],
});

const data = [
  {
    input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    output: [0, 0],
  },
  {
    input: [
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
      0.5,
    ],
    output: [0.5, 0.5],
  },
  {
    input: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    output: [1, 1],
  },
];



net.train(data);
document.body.innerHTML = brain.utilities.toSVG(net);

// net.train(data);
// document.body.innerHTML = net.run([
//   0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
//   0.5,
// ]);