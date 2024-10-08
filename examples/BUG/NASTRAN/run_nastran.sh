nastran="/msc/MSC_Nastran/2023.1/bin/nast20231"
option=""

run_nastran(){
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

move_outputs(){
    local file_in="$1"
    local basename="${file_in%.*}"
    local timestamp=$(date +"%m_%d_%y-%H_%M_%S")
    # Save the results of find into an array
    files=$(find . -maxdepth 1 -mindepth 1 -type f -name "$basename*" -not -name "${basename}.bdf")
    
    # Loop through the array
    for file in $files; do #"${files[@]}"
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
