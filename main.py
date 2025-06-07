import image_processor as ip
from graphite import Graphite

model = Graphite(model_path='graphite_test_val.pt')
ip.process('Screenshot.png', 'output.png')
print(model.predict('output.png'))
