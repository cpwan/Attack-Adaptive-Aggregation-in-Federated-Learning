yml=$1
scriptName=runxjobs_$yml.sh
>$scriptName
python -m runx.runx $yml.yml -i -n >>$scriptName;


numTotalJobs=$(wc -l < $scriptName)
echo "Run $numTotalJobs jobs in parallel?"

select yn in "Yes" "No"; do
    case $yn in
        Yes ) 
        parallel -j 48 --eta --delay 30 --limit 'python ./utils/allocateGPU.py 3300' <$scriptName;
        exit;;
        No ) exit;;
    esac
done
# 