import argparse, os
import subprocess
import logging

def main():
    parser = argparse.ArgumentParser(description='Pass arguments to another script')
    parser.add_argument("--gpus")
    parser.add_argument("--folderpath")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (default: Adam)')
    args = parser.parse_args()
    
    logFile = os.path.join(args.folderpath, 'runlog.log')
    logging.basicConfig(filename=logFile, level=logging.INFO)
    #Read until here

    # Execute the command and capture the output
    process = subprocess.Popen(['torchrun', '--nproc_per_node', args.gpus, '--nnodes', '1', 'multichannel.py', '--folderpath', args.folderpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()

    # Log the output
    logging.info(f"Standard output:\n{stdout.decode()}")
    if stderr:
        logging.warning(f"Standard error:\n{stderr.decode()}")

    return process.returncode

if __name__ == '__main__':
    main()
