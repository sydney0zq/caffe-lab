0. Start to train network

```
caffe train -solver lenet_auto_solver.protobuf
```


1. Resume from snapshot

```
caffe train -snapshot ./mnist/lenet_iter_2225.solverstate -solver lenet_auto_solver.protobuf
```



