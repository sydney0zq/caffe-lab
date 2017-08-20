while true; do
    sleep 600
    cat ../resnext_0819.log| grep Accuracy | grep Test > tmp_test.log
    cat ../resnext_0819.log| grep Accuracy | grep Train > tmp_train.log
    python plot_caffe_log.py
    mv ./tmp*.png ~/sync
    echo "Complete once..."
    date
done;

