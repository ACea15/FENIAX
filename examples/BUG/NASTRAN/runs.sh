#/bin/bash

# shopt -s expand_aliases
# Process command-line arguments
# if [ $# -lt 2 ]; then
#     echo "Usage: $0 <arg1> <arg2> [-o <option>] [-h]"
#     exit 1
# fi

option=""

# Process options using getopts
while getopts ":s:hc:" opt; do
    case $opt in
        s)
            s_option="$OPTARG" ;;
	
        c)
            option="$OPTARG" ;;
	
        h)
            echo "Usage: $0 <arg1> <arg2> [-o <option>] [-h]"
            echo "  -o <option>   Specify an option"
            echo "  -h            Display this help message"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument."
            exit 1
            ;;
    esac
done

if [ "$s_option" = "s103c" ]

then

    nastran XRF1-144run.bdf  scr=yes batch=no &
    
    pidn=$!

    wait $pidn

    if grep -q 'FATAL' XRF1-144run.f06

    then echo 'FATAL error in a'

    fi

    find . -maxdepth 1 -mindepth 1 -name 'XRF1-144run.*' -not -name 'XRF1-144run.bdf' -print0 | xargs -0 mv -t runs

fi


# Shift the processed options, so $1 and $2 refer to the positional arguments
#shift "$((OPTIND-1))"

# Your script logic using the arguments and options
#echo "arg1: $arg1"
#echo "arg2: $arg2"
