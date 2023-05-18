#!/bin/bash

echo "Bash version ${BASH_VERSION}..."
#python variables.py
#python loads.py
for i in {50..900..50}
do
    #python aero_loads.py "$i"
    #python ../../../../pyfem2nl_maint.py GolandWing confi_GW1
    #python ../../../../Utils/tool4main.py GolandWing confi_GW1
    echo "Welcome $i times"
done

function t1() {
    echo 'hello'
     }

function fact_elements() {
    var=("$1")
    mul=1
    for i in ${var[@]}
    do
        mul=$(($mul*i))
    done
    echo $mul
    }

function run_feminas() {

    varx=("$1")
    loadsx=("$2")
    aloadsx=("$3")
    #nvar=${#var[@]}
    #nloads=${#loads[*]}
    #naloads=${#aloads[@]}
    
    nvar=0
    var=()
    for j in ${varx[@]}
    do
        var+=($j)
        nvar=$((nvar+1))
    done    

    nloads=0
    loads=()
    for j in ${loadsx[@]}
    do
        loads+=($j)
        nloads=$((nloads+1))
    done    

    naloads=0
    aloads=()
    for j in ${aloadsx[@]}
    do
        aloads+=($j)
        naloads=$((naloads+1))
    done    
    #echo "loadsx ${loads[0]}" 
    nnvar=()
    for j in ${var[@]}
    do
        my_array=()
        IFS=',' read -ra my_array <<< $j              
        nj=${#my_array[@]}
        #echo ${my_array[@]}
        nnvar+=($nj)
    done    

    nnloads=()
    for j in ${loads[@]}
    do
        my_array=()
        IFS=',' read -ra my_array <<< $j              
        nj=${#my_array[@]}
        #echo ${my_array[@]}
        nnloads+=($nj)
    done
    
    nnaloads=()
    for j in ${aloads[@]}
    do
        my_array=()
        IFS=',' read -ra my_array <<< $j              
        nj=${#my_array[@]}
        #echo ${my_array[*]}
        #echo $nj
        nnaloads+=($nj)
    done
    #echo ${var[@]}
    #echo ${loads[@]}
    #echo ${aloads[@]}
    #echo start
    #echo ${nnaloads[1]}
    #echo ${nnvar[@]}
    #echo ${nnloads[@]}
    #echo ${nnaloads[@]}
    echo start
    perm=(${nnvar[@]} ${nnloads[@]} ${nnaloads[@]})
    #echo ${perm[@]} 
    ntotal=$(fact_elements "${perm[*]}")
    #echo ${ntotal[@]}
    for i in $(seq 0 $((ntotal-1)))
    do
        echo "seq ${i}" 
        if [ $nvar -gt 0 ]   
        then
            #echo $nvar
            vari=()
            for j in $(seq 0 $((nvar-1)))
            do
                IFS=',' read -ra my_array <<< ${var[j]}
                nj=$((i%nnvar[j]))
                nvari=${my_array[nj]}
                vari+=($nvari)
            done
            echo "variable ${vari[@]}"
            python variables.py $vari
        else
            echo "none"
            #python variables_trim.py                
        fi                
        if [ $nloads -gt 0 ]
        then   
            loadsi=()
            for j in $(seq 0 $((nloads-1)))
            do         
                IFS=',' read -ra my_array <<< ${loads[j]}              
                nj=$((i%nnloads[j]))
                nloadsi=${my_array[nj]}
                #echo ${my_array[@]}
                #echo ${nnloads[@]}
                #echo "loads $j"
                loadsi+=($nloadsi)
            done
            echo " loads ${loadsi[@]}"
            python loads.py $loadsi
        else
            echo "none"
            #python loads.py
        fi    
        if [ $naloads -gt 0 ]
        then   
            aloadsi=()
            for j in $(seq 0 $((naloads-1)))
            do         
                IFS=',' read -ra my_array <<< ${aloads[j]}              
                nj=$((i%nnaloads[j]))
                naloadsi=${my_array[nj]}
                aloadsi+=($naloadsi)
            done
            echo "aloads ${aloadsi[@]}"
            python aero_loads.py $aloadsi
        else
            echo "none"
            #python aero_loads.py         
        fi

        feminas_main.py XRF1-trim confi_trim

    done
}



# for vi in ${var[*]}
# do
#     IFS=',' read -ra my_array <<< $vi
#     for vj in ${my_array[@]}
#     do        

#         for li in ${loads[*]}
#         do
#             IFS=',' read -ra my_array2 <<< $li
#             for lj in ${my_array2[@]}
#             do        

#                 for ai in ${aloads[*]}
#                 do
#                     IFS=',' read -ra my_array3 <<< $ai
#                     for aj in ${my_array3[@]}
#                     do        

#                         python variables_trim.py "$vj"
#                         python loads.py  "$lj"
#                         python aero_loads.py "$aj"
#                         feminas_main.py XRF1-trim confi_trim
                        
#                     done
#                 done
                
                
#             done
#         done
        
       
#     done
# done


# for i in ${var[*]}
# do
#     IFS=',' read -ra my_array2 <<< $i
#     for j in ${my_array2[@]}
#     do        
#         echo $j
#         echo '#############'
#     done
# done


function write2config() {

    echo $#
    for i in $@
    do
        echo $i
    done
}


function sumOverArray() {
   val=("$1")
   #shift
   arr=("$2")
   echo ${val[@]}
   echo ${arr[@]}
   #for i in "${arr[@]}";
   #do
   #   sum=$((i + val))
   #   echo "sum: $sum"
   #done
}

