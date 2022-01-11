import unittest
import sys
import numpy as np
from copy import deepcopy
from evolution.Evolution import Evolution
from evolution.utils import StopConditions, FitnessFunctions, Selects, Mutations, Estimators

test_population = [
      [-0.34303962, -0.14006847],
      [-0.22943836,-0.37681416],
      [-0.12666971,1.72273752],
      [1.16532149,0.90285711],
      [-1.1224439 ,0.05341728],
      [-0.26837284,0.32076975],
      [-0.64500379,1.35992538],
      [1.02007229,2.23234931],
      [-0.31239878,-0.03610183],
      [ 1.27055288,-0.19951494],
      [1.10929522,1.61443861],
      [1.1496053 ,2.03025903],
      [-1.00837464,1.06989833],
      [-1.11482798,-1.76053901],
      [0.74976124,0.47700662],
      [ 0.24712949,-1.38777309],
      [1.99456762,0.07919687],
      [-2.63599853,1.44211043],
      [-1.09130137,1.87606951],
      [0.05983121,0.32046318],
      [-1.20779933,1.14272966],
      [0.31201385,1.32176869],
      [ 0.16251474,-1.32726464],
      [ 0.27301613,-0.07460142],
      [0.39499667,0.2190434 ],
      [-0.69058676,-0.66967541],
      [-1.32212335e+00,5.34071554e-06],
      [1.07217538,1.46864331],
      [-1.39521096,-0.79357373],
      [-0.77738137,-1.91395103],
      [-0.86574373,-1.95677275],
      [-1.79579397,-0.93618683],
      [0.43081035,2.16583805],
      [1.44530459,1.17192137],
      [-0.89319251,-0.81987624],
      [ 0.01205039,-0.98228016],
      [0.11522512,0.31550612],
      [-0.02763353,-0.19768746],
      [ 0.00464662,-1.45616461],
      [-1.89339341,0.22276381],
      [1.86931451,1.15461288],
      [ 0.28857485,-1.03201721],
      [ 1.14263316,-1.2507225 ],
      [1.30562067,0.69621048],
      [ 1.40103725,-0.11300773],
      [-0.84525407,-2.27660841],
      [-0.50413945,1.04664469],
      [-0.88300631,1.02615969],
      [-0.20170086,-1.64418394],
      [-0.69352956,-0.77420698],
      [-0.74316558,1.23777154],
      [-0.79594864,1.55159739],
      [-1.09258729,-0.37601563],
      [-0.48415103,-1.04253218],
      [ 2.04305884,-0.81051195],
      [2.27578463,1.36484032],
      [-1.21373163,-0.15893595],
      [ 0.15553162,-2.74978412],
      [0.52726681,0.40633855],
      [ 1.46438939,-0.20332035],
      [2.00379721,0.55220113],
      [-0.28580329,0.6034105 ],
      [0.47331189,0.21652363],
      [1.13344872,0.14180279],
      [-0.42689543,0.2788902 ],
      [-0.70572542,-0.17225903],
      [-0.5482615 ,0.47092229],
      [ 0.2464502 ,-1.33109879],
      [0.30688067,1.36046206],
      [-1.97789885,-1.49390571],
      [-1.7122581 ,-0.76221409],
      [-0.47019344,1.14115572],
      [-1.37362775,-1.36714046],
      [ 1.36865394,-2.91539562],
      [ 0.49766986,-1.77313913],
      [1.86787878,1.4695464 ],
      [0.3650815 ,0.96836702],
      [-1.99456772,0.81810192],
      [0.26485777,2.01304821],
      [ 0.39721759,-1.72497832],
      [ 0.95040644,-0.16756571],
      [-0.69995473,-0.15400078],
      [1.82288246,0.11976941],
      [-0.82148088,-1.59395945],
      [0.1696733 ,0.49631761],
      [0.60108974,1.66543799],
      [-0.47208856,-0.91387316],
      [0.48415256,0.55939045],
      [ 0.71389204,-0.85155658],
      [-0.27753626,0.37930812],
      [-1.54784995,-0.40500412],
      [0.39261525,0.7606199 ],
      [-0.53231261,0.84516125],
      [-1.99282355,0.16143246],
      [-0.35587276,-1.47559164],
      [-0.29982881,0.62938996],
      [ 0.56424029,-1.1336259 ],
      [0.20764485,0.46658267],
      [-0.33450687,-2.14302168],
      [0.04200341,0.63260354]
    ]

test_params = {
  'D': 2,
  'N': 100,
  'max_generations': 1,
  'fitness_func': FitnessFunctions.quadratic
}

eps = 0.00000001

class Test(unittest.TestCase):
  def testCentralPoint(self):
    evolution = Evolution(test_params)
    evolution.population = test_population
    proper_solution = np.mean(evolution.population, axis=0)
    test_solution = evolution.getCentralPoint(evolution.population)
    self.assertTrue(np.sum(np.absolute(np.subtract(proper_solution, test_solution))) < test_params['D'] * eps)
  
  def testBestPoint_quadratic(self):
    params = deepcopy(test_params)
    params['fitness_func'] = FitnessFunctions.quadratic
    evolution = Evolution(params)
    evolution.population = test_population
    # pop_values = list(map(lambda x: np.sum(np.absolute(x)), evolution.population))
    # proper_solution = evolution.population[np.argmin(pop_values)]
    proper_solution = [-0.02763353, -0.19768746]
    self.assertEqual(evolution.getBestPoint(evolution.population), proper_solution)

  def testBestPoint_eliptic(self):
    params = deepcopy(test_params)
    params['fitness_func'] = FitnessFunctions.eliptic
    evolution = Evolution(params)
    evolution.population = test_population
    proper_solution = [0.27301613, -0.07460142]
    self.assertEqual(evolution.getBestPoint(evolution.population), proper_solution)

  def testBestPoint_rastrigin(self):
    params = deepcopy(test_params)
    params['fitness_func'] = FitnessFunctions.rastrigin
    evolution = Evolution(params)
    evolution.population = test_population
    proper_solution = [0.00464662, -1.45616461]
    self.assertEqual(evolution.getBestPoint(evolution.population), proper_solution)

  def testBestPoint_rosenbrock(self):
    params = deepcopy(test_params)
    params['fitness_func'] = FitnessFunctions.rosenbrock
    evolution = Evolution(params)
    evolution.population = test_population
    proper_solution = [-0.34303962, -0.14006847]
    self.assertEqual(evolution.getBestPoint(evolution.population), proper_solution)

  def testBestPoint_triangular(self):
    params = deepcopy(test_params)
    params['fitness_func'] = FitnessFunctions.triangular
    evolution = Evolution(params)
    evolution.population = test_population
    proper_solution = [-0.02763353, -0.19768746]
    self.assertEqual(evolution.getBestPoint(evolution.population), proper_solution)

  def testSequence(self):
    params = deepcopy(test_params)
    params['fitness_func'] = FitnessFunctions.quadratic
    evolution = Evolution(params)
    evolution.population = test_population
    proper_solution = [
      [-0.03717479,  0.00925317],
      [-0.05137508,  0.03879507],
      [-0.02500137,  0.02447553],
      [-0.02686253,  0.05307614],
      [-0.05084844,  0.03941193],
      [-0.03056370,  0.05555212],
      [-0.04174068,  0.0323947 ],
      [-0.03310075,  0.05722269],
      [-0.05376357,  0.04187135],
      [-0.06698740,  0.02002093],
      [-0.07251849, -0.00382148],
      [-0.09628902,  0.00524246],
      [-0.11862542, -0.00781857],
      [-0.10744524, -0.02947245],
      [-0.10480499, -0.0048963 ],
      [-0.08257249, -0.01457863],
      [-0.07324902,  0.00854273],
      [-0.06069988,  0.02985696],
      [-0.08587667,  0.02348691],
      [-0.07733957,  0.0474059 ],
      [-0.08161704,  0.02283537],
      [-0.05991860,  0.03497489],
      [-0.03513777,  0.03335364],
      [-0.06149758,  0.03275827],
      [-0.07690275,  0.01194669],
      [-0.05961308,  0.03033452],
      [-0.03483226,  0.02773412],
      [-0.01185383,  0.03855533],
      [-0.03209214,  0.02281414],
      [-0.03955357,  0.04810925],
      [-0.06615980,  0.04708554],
      [-0.08265742,  0.02648325],
      [-0.07179236,  0.05031329],
      [-0.08183538,  0.02620695],
      [-0.08909376,  0.05274006],
      [-0.07821907,  0.02968072],
      [-0.07746203,  0.00322671],
      [-0.09682862,  0.02313066],
      [-0.07890974,  0.00507261],
      [-0.07689677,  0.03210961],
      [-0.05492486,  0.04587099],
      [-0.02962105,  0.05351295],
      [-0.02399602,  0.07987682],
      [-0.01310115,  0.05741983],
      [-0.03664975,  0.04601285],
      [-0.06394137,  0.05054618],
      [-0.08670550,  0.03476265],
      [-0.06931551,  0.01523178],
      [-0.07073786,  0.04352787],
      [-0.05755300,  0.02011132],
      [-0.06364665,  0.04826901],
      [-0.09353816,  0.05156038],
      [-0.10188022,  0.02429159],
      [-0.11068648, -0.00331431],
      [-0.09389692, -0.02569417],
      [-0.10146019,  0.00331482],
      [-0.10745962,  0.03355526],
      [-0.07921162,  0.03433549],
      [-0.11134887,  0.03990336],
      [-0.12782666,  0.06852602],
      [-0.11926749,  0.04171028],
      [-0.09120430,  0.04685505],
      [-0.07009935,  0.06966377],
      [-0.05836854,  0.04325888],
      [-0.02964024,  0.0549054 ],
      [-0.01665422,  0.08626076],
      [-0.05048077,  0.08462717],
      [-0.01799704,  0.08557292],
      [-0.04086858,  0.11485822],
      [-0.05149578,  0.1518542 ],
      [-0.03009466,  0.18272291],
      [-0.04372142,  0.15563173],
      [-0.02842260,  0.19382833],
      [-0.00976000,  0.16970489],
      [-0.01059886,  0.21401201],
      [-0.04903908,  0.22927512],
      [-0.02230792,  0.26673139],
      [-0.05587615,  0.25758899],
      [-0.07626212,  0.23472395],
      [-0.10294853,  0.21926364],
      [-0.07280969,  0.23883977],
      [-0.04778591,  0.2266249 ],
      [-0.01155431,  0.24777077],
      [ 0.00540302,  0.22532258],
      [ 0.02360341,  0.20169209],
      [-0.00997415,  0.18804899],
      [-0.01368683,  0.1562951 ],
      [-0.02779146,  0.13013952],
      [-0.06955007,  0.12294084],
      [-0.09474961,  0.09170068],
      [-0.06153503,  0.07298172],
      [-0.03753489,  0.03894546],
      [-0.09160133,  0.01643321],
      [-0.07191033,  0.07261141],
      [-0.03916658,  0.03125169],
      [ 0.02160803,  0.06551572],
      [-0.00179624,  0.00301812],
      [-0.02233873, -0.1027969 ],
      [ 0.12269130, -0.13614444],
      [-0.02763353, -0.19768746],
    ]
    test_solution = evolution.getSequence(evolution.population)
    self.assertTrue(np.sum(np.abs(np.subtract(proper_solution, test_solution))) < eps * test_params['D'] * test_params['N'])

  def testEstimator_quadratic(self):
    params = deepcopy(test_params)
    evolution = Evolution(test_params)
    evolution.population = test_population
    test_solution = evolution.estimate(evolution.getSequence(evolution.population))
    proper_solution = [-0.03660783,  0.11650104]
    self.assertTrue(np.sum(np.abs(np.subtract(proper_solution, test_solution))) < eps * test_params['D'])
