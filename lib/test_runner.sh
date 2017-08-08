#!/bin/zsh

# . ~/bin/common.sh
. ~/Environments/avians/bin/activate 
# test_files=('test_video_search.py' 'avians/util/test_arabic_nlp.py' 'avians/util/test_image.py')
test_files=('avians/web/test_main.py' 'avians/test_video_search.py')
sha1f=/tmp/nose-avians-sha1
if [[ -f $sha1f ]] ; then
    AVIANS_CURRENT_SHA1=$(cat $sha1f)
else 
    echo "Cannot find Avians Current Sha1" 
    AVIANS_CURRENT_SHA1=
fi
git pull -q office
NEW_SHA1=$(git log -n 1 | head -n 1)
echo "$NEW_SHA1"
if [[ $NEW_SHA1 != $AVIANS_CURRENT_SHA1 ]] ; then
    echo $NEW_SHA1 > $sha1f
    SLEEP_COUNTER=30
    for test_file in ${test_files} ; do

        print "Testing File: ${test_file}"
        test_funcs=$(grep -o '^def test_[0-9A-Za-z_]\+' $test_file | grep -o 'test_[0-9A-Za-z_]\+')
        for tf in ${(@f)test_funcs} ; do
            echo ${test_file}:${tf}
            outfile=/tmp/nose-avians-${NEW_SHA1[8,14]}-${test_file:t}:${tf}-${RANDOM}.txt
            if false ; then
                echo "Testing with Profiling"
                profiling_dat=/tmp/nose-cprof-${NEW_SHA1[8,14]}-${test_file:t}:${tf}.dat
                profiling_out=/tmp/nose-cprof-${NEW_SHA1[8,14]}-${test_file:t}:${tf}.txt
                nosetests --nocapture --with-cprofile --cprofile-stats-file=${profiling_dat} ${test_file}:${tf} 2>&1 | tee $outfile

                python3 > ${profiling_out} <<EOF
import pstats
s = pstats.Stats("${profiling_dat}")
s.sort_stats('tottime')
s.print_stats()
EOF

                echo "\nPROFILING OUTPUT\n" >> "${outfile}"
                head -n 100 $profiling_out >> ${outfile}
            else

                echo "Testing without Profiling"
                nosetests --nocapture ${test_file}:${tf} 2>&1 | tee $outfile
            fi

            if [[ -n "$(grep '\(ERROR\|FAIL\)' $outfile)" ]] ; then
                filename="$HOME/org/INBOX/$(date "+%WW%u %H%M") avians:${NEW_SHA1[8,14]}:${test_file:t}:${tf} - $(grep '\(ERROR\|FAIL\)' $outfile | tr -c 'a-zA-Z0-9:()[]+-' '_').cres"
                mv $outfile "${filename}"
            fi
        done
        ntfy send "$test_file tests done"
    done
else
    echo "No Changes in the Repository"
    SLEEP_COUNTER=$(($RANDOM %  30))
fi
echo "Sleeping for $SLEEP_COUNTER seconds"
sleep $SLEEP_COUNTER
