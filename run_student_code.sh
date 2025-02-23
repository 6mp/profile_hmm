s=/autograder/source
for x in $s/Data/PF*hmm; do
    timeout 120 $s/profile_hmm -m $x -q $s/Data/queries.fas -o $s/Data/`basename $x .hmm`_student_output.txt > $s/Data/`basename $x .hmm`_student_error.log 2>&1
    if [ $? -eq 124 ]; then
        echo "Time out after 120 seconds!" >> $s/Data/`basename $x .hmm`_student_error.log
    fi
done

