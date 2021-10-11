// boilerplate, żeby działało
function cloneDeep(a) {
  return JSON.parse(JSON.stringify(a));
}
function getPopulation(size) {
  return [
    [0.1, 0.1],
    [0.2, 0.3],
    [0.2, 0.15],
    [0.1, -0.1],
    [0.1, -0.2],
    [-0.15, 0.2],
  ];
}
// koniec boilerplate'u

const D = 2; // ilość wymiarów
const N = 100; // ilość punktów w populacji

// Definicja funkcji celu dla D-wymiarowej przestrzeni
function evaluate(point) {
  const sum = point.reduce((acc, attribute) => acc + attribute * attribute, 0);
  const dimensions = point.length;
  return sum / dimensions;
}

// sortowanie według funkcji celu
function descendingByValue(a, b) {
  if (this.evaluate(a) > this.evaluate(b)) {
    return -1;
  } else if (this.evaluate(a) === this.evaluate(b)) {
    return 0;
  } else {
    return 1;
  }
}

function getCentralPoint(points) {
  const point = Array(D);
  for (let i = 0; i < D; i++) {
    point[i] =
      points.reduce((sum, point) => sum + point[i], 0) /
      Math.max(points.length);
  }
  return point;
}
function calculateSequence() {
  // tu zdefiniować sposób generacji populacji
  const population = getPopulation((size = N));

  const tempPopulation = cloneDeep(population).sort(descendingByValue);
  const centroidSequence = [];
  while (tempPopulation.length) {
    const centralPoint = getCentralPoint(tempPopulation);
    centroidSequence.push(centralPoint);
    tempPopulation.pop();
  }
  return centroidSequence;
}

const centroidSequence = calculateSequence();
console.log(centroidSequence.map((point) => evaluate(point)));
