python mian.py -m nesterov --ablation -r 0.001 && python main.py -m nesterov --ablation -r 0.01 && python main.py -m nesterov --ablation -r 0.1 &&\
python main.py -m adagrad --ablation -r 0.001 && python main.py -m adagrad --ablation -r 0.01 && python main.py -m adagrad --ablation -r 0.1 &&\
python main.py -m adam --ablation -r 0.001 && python main.py -m adam --ablation -r 0.01 && python main.py -m adam --ablation -r 0.1 &&\
python main.py -m rmsprop --ablation -r 0.001 && python main.py -m rmsprop --ablation -r 0.01 && python main.py -m rmsprop --ablation -r 0.1 &&\
python main.py -m adadelta --ablation -r 0.001 && python main.py -m adadelta --ablation -r 0.01 && python main.py -m adadelta --ablation -r 0.1 
