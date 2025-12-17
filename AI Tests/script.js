const net = new brain.NeuralNetwork({
  hiddenLayers: [8,8,8],
});


const data = [];


let train_count = 1000;
let test_count = 10000;
let correct = 0;

xin = document.getElementById("x");
yin = document.getElementById("y");
outAI = document.getElementById("outAI");
outCalc = document.getElementById("outCalc");
percentDiv = document.getElementById("Percent");

console.log("Createing Training data")
for (let i = 0; i < train_count; i++) {
  x = Math.random();
  y = x;
  value = x + y;

  data.push({
    input: [x, y],
    output: [value],
  });
//   console.log(`${i}. ${x} + ${y} = ${value}`);
}
console.log("Training")
net.train(data);

console.log("Testing")
for (let i = 0; i < test_count; i++) {
  x = Math.round(Math.random() * 0.5 * 100) / 100;
  y = Math.round(Math.random() * 0.5 * 100) / 100;

  calculated = Math.round(net.run([x, y])[0] * 100) / 100;

  calculator = Math.round((x + y) * 100) / 100;
  if (calculated == calculator) {
    correct++;
  }

//   console.log(
//     `${i}. X = ${x}, Y = ${y} and AI = ${calculated}, Calculator = ${calculator}`
//   );
}

percentDiv.innerHTML = `The Moddel is to ${Math.round(correct / test_count * 100 * 100) / 100}% correct.`
console.log("Startup done")
console.log(
  `The Moddel is to ${
    Math.round((correct / test_count) * 100 * 100) / 100
  }% correct.`
);



document.getElementById("calc").addEventListener("click", () => {
  calculated = net.run([xin.value / 100, yin.value / 100])[0];
  console.log(calculated);
  calculated = Math.round(calculated * 100);
  console.log(calculated);
  outAI.innerHTML = "AI        : <b>" + calculated + "</b>";
  calculated = xin.value / 1 + yin.value / 1;
  outCalc.innerHTML = "Calculator: <b>" + calculated + "</b>";
});