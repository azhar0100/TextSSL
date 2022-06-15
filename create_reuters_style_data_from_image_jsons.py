import argparse
import itertools
import pickle
from tqdm import tqdm
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Take an imagejson pkl and make a dataset that can be input to the rest of the pipeline')
    parser.add_argument('input_imagejson_pkl')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    input_imagejson_pkl = args.input_imagejson_pkl
    output_folder = args.output_folder

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    with open(input_imagejson_pkl, 'rb') as f:
        pkl_data = pickle.load(f)


    for data_item in tqdm(pkl_data):

        # Gather all the text, the 'sentences' in block_texts
        block_nums = list(set(data_item['block']))
        block_nums.sort()
        block_texts = []
        for block_num in block_nums:
            block_text = " ".join([w for w,b in zip(data_item['words'],data_item['block']) if b == block_num])
            block_text = block_text.replace('.','') + '. '
            block_texts.append( block_text )
        
        final_text = '\n'.join(block_texts)

        # Find the ouput filename
        filename = Path(data_item['image_path'])
        output_name = filename.stem + '.txt'
        output_path = output_folder / output_name
        with open(output_path, 'w') as f:
            f.write(final_text)
