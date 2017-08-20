while true; do
    sleep 60
    cat ../resnext_0820.log| grep Accuracy | grep Test > tmp_test.log
    cat ../resnext_0820.log| grep Accuracy | grep Train > tmp_train.log
    python plot_caffe_log.py
    mv ./tmp*.png ~/sync
    echo "Complete once..."
    date
done;

