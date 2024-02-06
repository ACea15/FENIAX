#/bin/zsh

nastran="/msc/MSC_Nastran/2023.1/bin/nast20231"
option=""

# Process options using getopts
while getopts ":s:hc:o" opt; do
    case $opt in
        s)
            s_option="$OPTARG" ;;
	
        c)
            option="$OPTARG" ;;

	o)
	    output="true" ;;
	
        h)
            echo "Usage: runs.sh  [-s <solution>] [-h]"
            echo "  -h Display this help message"	    
            echo "  -s <solution>   Specify a solution among these:"
            echo "     <103cao> Modal solution, outputs eigenvectors in op2,"
	    echo "              ASET model, clamped"
            echo "     <103eao> Modal solution, outputs eigenvectors in op2,"
	    echo "              ASET model, free"
	    
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

function run_nastran(){
    local file_in="$1"
    local file_out="$(echo $file_in | sed -e 's/bdf/f06/')"
    # ANOTHER WAY:
    # Get the basename of the file without the extension
    # local basename="${filename%.*}"
    # Replace the extension with the new suffix
    # local new_filename="${basename}.f06"    
    $nastran $file_in  scr=yes batch=no &
    pidn=$!
    wait $pidn
    if grep -q 'FATAL' $file_out

    then echo 'FATAL errors!'
	 if [ "$output" = "true" ]
	 then
	     grep 'FATAL' $file_out

	 fi
    fi

}

function move_outputs(){
    local file_in="$1"
    local basename="${file_in%.*}"
    local timestamp=$(date +"%m_%d_%y-%H_%M_%S")
    # Save the results of find into an array
    files=($(find . -maxdepth 1 -mindepth 1 -type f -name "$basename*" -not -name "${basename}.bdf"))

    # Loop through the array
    for file in "${files[@]}"; do
	# Get the basename of the file without the directory path
	fileonly="${file##*/}"
	
	# Get the file extension
	extension="${file##*.}"

	# Get the filename without the extension
	filename="${fileonly%.*}"
	
	# Move the file to the destination directory with the modified name
	mv "$file" "results_runs/${filename}-${timestamp}.${extension}"
    done
    
}

if [ "$s_option" = "103cao" ]

then
    run_nastran "BUG_103cao.bdf"
    move_outputs "BUG_103cao.bdf"
fi

if [ "$s_option" = "103cfo" ]

then
    run_nastran "BUG_103cfo.bdf"
    move_outputs "BUG_103cfo.bdf"    
fi

if [ "$s_option" = "103eao" ]

then
    run_nastran "BUG_103eao.bdf"
    move_outputs "BUG_103eao.bdf"    
fi

if [ "$s_option" = "103efo" ]

then
    run_nastran "BUG_103efo.bdf"
    move_outputs "BUG_103efo.bdf"    
fi

# if [ "$s_option" = "103cao" ]

# then

#     $nastran BUG_103cao.bdf  scr=yes batch=no &
    
#     pidn=$!

#     wait $pidn
#     fatal="$(grep 'FATAL' BUG_103cao.f06)"
#     if grep -q 'FATAL' BUG_103cao.f06

#     then echo 'FATAL errors:'
# 	 if [ "$output" = "true" ]
# 	 then
# 	     grep 'FATAL' BUG_103cao.f06

# 	 fi

# 	 # find . -maxdepth 1 -mindepth 1 -name '144run.*' -not -name '144run.bdf' -print0 | xargs -0 mv -t runs

#     fi
# fi

# Shift the processed options, so $1 and $2 refer to the positional arguments
#shift "$((OPTIND-1))"

# Your script logic using the arguments and options
#echo "arg1: $arg1"
#echo "arg2: $arg2"
