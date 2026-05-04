import argparse
from config import Config
from train import train_lstm_predictor, train_dqn_jammer, train_hybrid
from evaluate import evaluate_strategies
from visualize import visualize_jamming

def main():
    parser = argparse.ArgumentParser(description='Jammer Optimization with Neural Networks')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'visualize', 'all'],
                        default='all')
    parser.add_argument('--algorithm', default='markov',
                        choices=['pseudo_random', 'chaotic', 'markov', 'adaptive'])
    args = parser.parse_args()
    
    config = Config()
    config.FH_ALGORITHM = args.algorithm
    
    if args.mode in ['train', 'all']:
        train_lstm_predictor(config)
        train_dqn_jammer(config)
        train_hybrid(config)
    
    if args.mode in ['evaluate', 'all']:
        evaluate_strategies(config)
    
    if args.mode in ['visualize', 'all']:
        visualize_jamming(config)

if __name__ == "__main__":
    main()