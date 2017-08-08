
import os
import shutil
import argparse
import numpy as np

def stem_to_letter(input_dir, output_dir):

    input_classes = os.listdir(input_dir)
    output_classes = {}
    for ic in input_classes:
        icf = os.listdir(os.path.join(input_dir, ic))
        len_class = "len-" + str(len(ic))
        if len_class not in output_classes: 
            output_classes[len_class] = [os.path.join(ic, f) for f in icf]
        else: 
            output_classes[len_class] += [os.path.join(ic, f) for f in icf]
        for i, l in enumerate(ic):
            new_cls = "{}-{}".format(i+1, l)
            if new_cls not in output_classes: 
                output_classes[new_cls] = [os.path.join(ic, f) for f in icf]
            else: 
                output_classes[new_cls] += [os.path.join(ic, f) for f in icf]

    for outd, outf in output_classes.items():
        oo = os.path.join(output_dir, outd)
        os.makedirs(oo)
        for f in outf:
            iif = os.path.join(input_dir, f)
            shutil.copy(iif, oo)
    

def main(): 
    parser = argparse.ArgumentParser(description="""
    
    Divides the letter group dataset into letter classes. 

    From a directory like: 
    d/a/
    d/ab/
    d/abc/
    d/ae/
    d/ba/

    it produces a directory structure

    d/1a
    d/1b
    d/2a
    d/2b
    d/2e
    d/3c

    to index the letters for each group. 

    """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input_dir', help="Document to produce the dataset")
    parser.add_argument('--output_dir', help="Number of components to produce", 
                        default='/tmp/letter-classes-{}'.format(np.random.randint(10000)))
    args = vars(parser.parse_args())
    print(args)
    stem_to_letter(args['input_dir'], args['output_dir'])

if __name__ == '__main__': 
    main()

