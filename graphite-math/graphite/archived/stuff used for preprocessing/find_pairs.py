import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

VARIANT_SUFFIXES = ['', '_var1', '_var2', '_var3', '_var4']

def _extract_latex(inkml_path):
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    namespace = {'ns': 'http://www.w3.org/2003/InkML'}

    for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
        if annotation.attrib.get('type') == 'truth':
            return annotation.text.strip()

    for annotation in root.findall('.//annotation'):
        if annotation.attrib.get('type') == 'truth':
            return annotation.text.strip()

    return None

def find_pairs_complete(image_folder: str, inkml_folder: str):
    image_files = [f for f in os.listdir(image_folder) if '.png' in f]
    pairs = []
    for img in image_files:
        inkml_file = os.path.join(inkml_folder, f'{img[:-4]}.inkml')
        try:
            pairs.append({
                'img': os.path.join(image_folder, img),
                'latex': _extract_latex(inkml_file),
                'type': 'complete'
            })
        except FileNotFoundError:
            pass
    return pairs

def all_roots(pair_list, is_complete):
    new_list = []
    for folder, dest in pair_list:
        for var in VARIANT_SUFFIXES:
            if is_complete:
                new_list.append((folder + var, dest))
            else:
                parts = folder.split('/')
                new_folder = '/'.join(parts[:-1]) + var + '/' + parts[-1]
                new_list.append((new_folder, dest))
    return new_list

def get_symbol_dicts(symbol_tuple):
    folder_path, latex = symbol_tuple
    result = []
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        # Folder is missing, skip by returning empty list
        return []
    for filename in files:
        if filename.lower().endswith('.png'):
            result.append({
                'img': os.path.join(folder_path, filename),
                'latex': latex,
                'type': 'symbol'
            })
    return result

complete_pairs = [
    # FIRST BATCH
    ('./CLEANED_DATA/2019_test', './DATA/INKML/test/CROHME2019_test'),
    ('./CLEANED_DATA/2023_test', './DATA/INKML/test/CROHME2023_test'),
    ('./CLEANED_DATA/2019_train', './DATA/INKML/train/CROHME2019'),
    ('./CLEANED_DATA/2023_train', './DATA/INKML/train/CROHME2023_train'),
    ('./CLEANED_DATA/2016_val', './DATA/INKML/val/CROHME2016_test'),
    ('./CLEANED_DATA/2023_val', './DATA/INKML/val/CROHME2023_val'),

    # NEXT BATCHES
    ('./CLEANED_DATA/2011_test', './2ND_BATCH_DATA/CROHME_test_2011'),
    ('./CLEANED_DATA/2011_train', './2ND_BATCH_DATA/CROHME_training_2011'),
    ('./CLEANED_DATA/2014_test', './2ND_BATCH_DATA/MatricesTest2014'),
    ('./CLEANED_DATA/2014_train', './2ND_BATCH_DATA/MatricesTrain2014'),
    ('./CLEANED_DATA/2012_test', './2ND_BATCH_DATA/testData_2012'),
    ('./CLEANED_DATA/2013_test', './2ND_BATCH_DATA/TestINKML_2013'),
    ('./CLEANED_DATA/2012_train_1', './2ND_BATCH_DATA/trainData_2012_part1'),
    ('./CLEANED_DATA/2012_train_2', './2ND_BATCH_DATA/trainData_2012_part2'),
    ('./CLEANED_DATA/2013_train', './2ND_BATCH_DATA/TrainINKML_2013')
]

img_latex_pairs = []

print('Adding complete pairs...')
for image, latex in tqdm(all_roots(complete_pairs, is_complete=True)):
    img_latex_pairs += find_pairs_complete(image, latex)

symbol_pairs = [
    ('./CLEANED_DATA/symbols1/!', '!'),
    ('./CLEANED_DATA/symbols1/(', '('),
    ('./CLEANED_DATA/symbols1/)', ')'),
    ('./CLEANED_DATA/symbols1/+', '+'),
    ('./CLEANED_DATA/symbols1/,', ','),
    ('./CLEANED_DATA/symbols1/-', '-'),
    ('./CLEANED_DATA/symbols1/0', '0'),
    ('./CLEANED_DATA/symbols1/1', '1'),
    ('./CLEANED_DATA/symbols1/2', '2'),
    ('./CLEANED_DATA/symbols1/3', '3'),
    ('./CLEANED_DATA/symbols1/4', '4'),
    ('./CLEANED_DATA/symbols1/5', '5'),
    ('./CLEANED_DATA/symbols1/6', '6'),
    ('./CLEANED_DATA/symbols1/7', '7'),
    ('./CLEANED_DATA/symbols1/8', '8'),
    ('./CLEANED_DATA/symbols1/9', '9'),
    ('./CLEANED_DATA/symbols1=', '='),
    ('./CLEANED_DATA/symbols1/[', '['),
    ('./CLEANED_DATA/symbols1/]', ']'),
    ('./CLEANED_DATA/symbols1/a_lower', 'a'),
    ('./CLEANED_DATA/symbols1/a_upper', 'A'),
    ('./CLEANED_DATA/symbols1/alpha', '\\alpha'),
    ('./CLEANED_DATA/symbols1/b_lower', 'b'),
    ('./CLEANED_DATA/symbols1/b_upper', 'B'),
    ('./CLEANED_DATA/symbols1/beta', '\\beta'),
    ('./CLEANED_DATA/symbols1/c_lower', 'c'),
    ('./CLEANED_DATA/symbols1/c_upper', 'C'),
    ('./CLEANED_DATA/symbols1/d_lower', 'd'),
    ('./CLEANED_DATA/symbols1/d_upper', 'D'),
    ('./CLEANED_DATA/symbols1/Delta', '\\Delta'),
    ('./CLEANED_DATA/symbols1/div', '\\div'),
    ('./CLEANED_DATA/symbols1/e_lower', 'e'),
    ('./CLEANED_DATA/symbols1/e_upper', 'E'),
    ('./CLEANED_DATA/symbols1/exists', '\\exists'),
    ('./CLEANED_DATA/symbols1/f_lower', 'f'),
    ('./CLEANED_DATA/symbols1/f_upper', 'F'),
    ('./CLEANED_DATA/symbols1/forall', '\\forall'),
    ('./CLEANED_DATA/symbols1/forward_slash', '/'),
    ('./CLEANED_DATA/symbols1/g_lower', 'g'),
    ('./CLEANED_DATA/symbols1/g_upper', 'G'),
    ('./CLEANED_DATA/symbols1/gamma', '\\gamma'),
    ('./CLEANED_DATA/symbols1/geq', '\\geq'),
    ('./CLEANED_DATA/symbols1/gt', '\\gt'),
    ('./CLEANED_DATA/symbols1/h_lower', 'h'),
    ('./CLEANED_DATA/symbols1/h_upper', 'H'),
    ('./CLEANED_DATA/symbols1/i_lower', 'i'),
    ('./CLEANED_DATA/symbols1/i_upper', 'I'),
    ('./CLEANED_DATA/symbols1/in', '\\in'),
    ('./CLEANED_DATA/symbols1/infty', '\\infty'),
    ('./CLEANED_DATA/symbols1/int', '\\int'),
    ('./CLEANED_DATA/symbols1/j_lower', 'j'),
    ('./CLEANED_DATA/symbols1/j_upper', 'J'),
    ('./CLEANED_DATA/symbols1/k_lower', 'k'),
    ('./CLEANED_DATA/symbols1/k_upper', 'K'),
    ('./CLEANED_DATA/symbols1/l_lower', 'l'),
    ('./CLEANED_DATA/symbols1/l_upper', 'L'),
    ('./CLEANED_DATA/symbols1/lambda', '\\lambda'),
    ('./CLEANED_DATA/symbols1/ldots', '\\ldots'),
    ('./CLEANED_DATA/symbols1/leq', '\\leq'),
    ('./CLEANED_DATA/symbols1/lim', '\\lim'),
    ('./CLEANED_DATA/symbols1/log', '\\log'),
    ('./CLEANED_DATA/symbols1/lt', '\\lt'),
    ('./CLEANED_DATA/symbols1/m_lower', 'm'),
    ('./CLEANED_DATA/symbols1/m_upper', 'M'),
    ('./CLEANED_DATA/symbols1/mu', '\\mu'),
    ('./CLEANED_DATA/symbols1/n_lower', 'n'),
    ('./CLEANED_DATA/symbols1/n_upper', 'N'),
    ('./CLEANED_DATA/symbols1/neq', '\\neq'),
    ('./CLEANED_DATA/symbols1/o_lower', 'o'),
    ('./CLEANED_DATA/symbols1/o_upper', 'O'),
    ('./CLEANED_DATA/symbols1/p_lower', 'p'),
    ('./CLEANED_DATA/symbols1/p_upper', 'P'),
    ('./CLEANED_DATA/symbols1/phi', '\\phi'),
    ('./CLEANED_DATA/symbols1/pi', '\\pi'),
    ('./CLEANED_DATA/symbols1/pm', '\\pm'),
    ('./CLEANED_DATA/symbols1/prime', '\\prime'),
    ('./CLEANED_DATA/symbols1/q_lower', 'q'),
    ('./CLEANED_DATA/symbols1/q_upper', 'Q'),
    ('./CLEANED_DATA/symbols1/r_lower', 'r'),
    ('./CLEANED_DATA/symbols1/r_upper', 'R'),
    ('./CLEANED_DATA/symbols1/rightarrow', '\\rightarrow'),
    ('./CLEANED_DATA/symbols1/s_lower', 's'),
    ('./CLEANED_DATA/symbols1/s_upper', 'S'),
    ('./CLEANED_DATA/symbols1/sigma', '\\sigma'),
    ('./CLEANED_DATA/symbols1/sin', '\\sin'),
    ('./CLEANED_DATA/symbols1/sqrt', '\\sqrt'),
    ('./CLEANED_DATA/symbols1/sum', '\\sum'),
    ('./CLEANED_DATA/symbols1/t_lower', 't'),
    ('./CLEANED_DATA/symbols1/t_upper', 'T'),
    ('./CLEANED_DATA/symbols1/tan', '\\tan'),
    ('./CLEANED_DATA/symbols1/theta', '\\theta'),
    ('./CLEANED_DATA/symbols1/times', '\\times'),
    ('./CLEANED_DATA/symbols1/v_lower', 'v'),
    ('./CLEANED_DATA/symbols1/v_upper', 'V'),
    ('./CLEANED_DATA/symbols1/w_lower', 'w'),
    ('./CLEANED_DATA/symbols1/w_upper', 'W'),
    ('./CLEANED_DATA/symbols1/x_lower', 'x'),
    ('./CLEANED_DATA/symbols1/x_upper', 'X'),
    ('./CLEANED_DATA/symbols1/y_lower', 'y'),
    ('./CLEANED_DATA/symbols1/y_upper', 'Y'),
    ('./CLEANED_DATA/symbols1/z_lower', 'z'),
    ('./CLEANED_DATA/symbols1/z_upper', 'Z'),
    ('./CLEANED_DATA/symbols1/{', '{'),
    ('./CLEANED_DATA/symbols1/|', '\\mid'),
    ('./CLEANED_DATA/symbols1/}', '}'),
    ('./CLEANED_DATA/symbols2/0', '0'),
    ('./CLEANED_DATA/symbols2/1', '1'),
    ('./CLEANED_DATA/symbols2/2', '2'),
    ('./CLEANED_DATA/symbols2/3', '3'),
    ('./CLEANED_DATA/symbols2/4', '4'),
    ('./CLEANED_DATA/symbols2/5', '5'),
    ('./CLEANED_DATA/symbols2/6', '6'),
    ('./CLEANED_DATA/symbols2/7', '7'),
    ('./CLEANED_DATA/symbols2/8', '8'),
    ('./CLEANED_DATA/symbols2/9', '9'),
    ('./CLEANED_DATA/symbols2/add', '+'),
    ('./CLEANED_DATA/symbols2/dec', '.'),
    ('./CLEANED_DATA/symbols2/div', '\\div'),
    ('./CLEANED_DATA/symbols2/eq', '='),
    ('./CLEANED_DATA/symbols2/mul', '\\times'),
    ('./CLEANED_DATA/symbols2/sub', '-'),
    ('./CLEANED_DATA/symbols2/x', 'x'),
    ('./CLEANED_DATA/symbols2/y', 'y'),
    ('./CLEANED_DATA/symbols2/z', 'z'),
    ('./CLEANED_DATA/symbols3/divide', '\\div'),
    ('./CLEANED_DATA/symbols3/eight', '8'),
    ('./CLEANED_DATA/symbols3/equals', '='),
    ('./CLEANED_DATA/symbols3/five', '5'),
    ('./CLEANED_DATA/symbols3/four', '4'),
    ('./CLEANED_DATA/symbols3/minus', '-'),
    ('./CLEANED_DATA/symbols3/mult', '\\times'),
    ('./CLEANED_DATA/symbols3/nine', '9'),
    ('./CLEANED_DATA/symbols3/one', '1'),
    ('./CLEANED_DATA/symbols3/plus', '+'),
    ('./CLEANED_DATA/symbols3/point', '.'),
    ('./CLEANED_DATA/symbols3/seven', '7'),
    ('./CLEANED_DATA/symbols3/six', '6'),
    ('./CLEANED_DATA/symbols3/three', '3'),
    ('./CLEANED_DATA/symbols3/two', '2'),
    ('./CLEANED_DATA/symbols3/zero', '0'),
    ('./CLEANED_DATA/symbols4/decimal', '.'),
    ('./CLEANED_DATA/symbols4/div', '\\div'),
    ('./CLEANED_DATA/symbols4/eight', '8'),
    ('./CLEANED_DATA/symbols4/equal', '='),
    ('./CLEANED_DATA/symbols4/five', '5'),
    ('./CLEANED_DATA/symbols4/four', '4'),
    ('./CLEANED_DATA/symbols4/minus', '-'),
    ('./CLEANED_DATA/symbols4/nine', '9'),
    ('./CLEANED_DATA/symbols4/one', '1'),
    ('./CLEANED_DATA/symbols4/plus', '+'),
    ('./CLEANED_DATA/symbols4/seven', '7'),
    ('./CLEANED_DATA/symbols4/six', '6'),
    ('./CLEANED_DATA/symbols4/three', '3'),
    ('./CLEANED_DATA/symbols4/times', '\\times'),
    ('./CLEANED_DATA/symbols4/two', '2'),
    ('./CLEANED_DATA/symbols4/zero', '0'),
    ('./CLEANED_DATA/symbols5/add', '+'),
    ('./CLEANED_DATA/symbols5/divide', '\\div'),
    ('./CLEANED_DATA/symbols5/eight', '8'),
    ('./CLEANED_DATA/symbols5/five', '5'),
    ('./CLEANED_DATA/symbols5/four', '4'),
    ('./CLEANED_DATA/symbols5/multiply', '\\times'),
    ('./CLEANED_DATA/symbols5/nine', '9'),
    ('./CLEANED_DATA/symbols5/one', '1'),
    ('./CLEANED_DATA/symbols5/seven', '7'),
    ('./CLEANED_DATA/symbols5/six', '6'),
    ('./CLEANED_DATA/symbols5/subtract', '-'),
    ('./CLEANED_DATA/symbols5/three', '3'),
    ('./CLEANED_DATA/symbols5/two', '2'),
    ('./CLEANED_DATA/symbols5/zero', '0'),
    ('./CLEANED_DATA/symbols6/add', '+'),
    ('./CLEANED_DATA/symbols6/divide', '\\div'),
    ('./CLEANED_DATA/symbols6/eight', '8'),
    ('./CLEANED_DATA/symbols6/five', '5'),
    ('./CLEANED_DATA/symbols6/four', '4'),
    ('./CLEANED_DATA/symbols6/multiply', '\\times'),
    ('./CLEANED_DATA/symbols6/nine', '9'),
    ('./CLEANED_DATA/symbols6/one', '1'),
    ('./CLEANED_DATA/symbols6/seven', '7'),
    ('./CLEANED_DATA/symbols6/six', '6'),
    ('./CLEANED_DATA/symbols6/subtract', '-'),
    ('./CLEANED_DATA/symbols6/three', '3'),
    ('./CLEANED_DATA/symbols6/two', '2'),
    ('./CLEANED_DATA/symbols6/zero', '0'),
    ('./CLEANED_DATA/symbols7/add', '+'),
    ('./CLEANED_DATA/symbols7/divide', '\\div'),
    ('./CLEANED_DATA/symbols7/eight', '8'),
    ('./CLEANED_DATA/symbols7/five', '5'),
    ('./CLEANED_DATA/symbols7/four', '4'),
    ('./CLEANED_DATA/symbols7/multiply', '\\times'),
    ('./CLEANED_DATA/symbols7/nine', '9'),
    ('./CLEANED_DATA/symbols7/one', '1'),
    ('./CLEANED_DATA/symbols7/seven', '7'),
    ('./CLEANED_DATA/symbols7/six', '6'),
    ('./CLEANED_DATA/symbols7/subtract', '-'),
    ('./CLEANED_DATA/symbols7/three', '3'),
    ('./CLEANED_DATA/symbols7/two', '2'),
    ('./CLEANED_DATA/symbols7/zero', '0'),
    ('./CLEANED_DATA/symbols8/(', '('),
    ('./CLEANED_DATA/symbols8/)', ')'),
    ('./CLEANED_DATA/symbols8/+', '+'),
    ('./CLEANED_DATA/symbols8/-', '-'),
    ('./CLEANED_DATA/symbols8/div', '\\div'),
    ('./CLEANED_DATA/symbols8/mult', '\\times')
]

print('Adding symbol pairs')
for pair in tqdm(all_roots(symbol_pairs, is_complete=False)):
    img_latex_pairs += get_symbol_dicts(pair)

with open('pairs.json', 'w') as f:
    json.dump({
        'pairs': img_latex_pairs
    }, f, indent=4)
