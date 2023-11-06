chmod +x run_hjb.sh

for i in {0..0}
    do
    nohup ./run_hjb.sh --device 'cpu' --par_idx $i &>4q_100cutoff_${i}.out &
    done

