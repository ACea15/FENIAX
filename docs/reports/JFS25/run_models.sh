#!/usr/bin/env bash

Manoeuvre="true"
Montecarlo="true"
MontecarloADt="true"
MontecarloADtjac="true"
MontecarloADtfd="true"    
MontecarloADfem="true"
Gust="true"
Forager="true"

pyenv activate feniax
pathBUG="../../../../FENIAXexamples/BUG/"
current_dir=$(pwd)
cd $pathBUG

if [ "$Manoeuvre" = "true" ]; then
    echo "RUNNING MANOEUVRE"
    python settings_manoeuvre1shard.py $current_dir
fi

if [ "$Montecarlo" = "true" ]; then
    echo "RUNNING MONTECARLO"
    python settings_DiscreteMC1high.py $current_dir
    python settings_DiscreteMC1small.py $current_dir
    python settings_DiscreteMC1vsmall.py $current_dir
fi

if [ "$MontecarloADt" = "true" ]; then
    echo "RUNNING MONTECARLOADT"
    # python settings_ADDiscreteLoadsMC_validation.py $current_dir
    python settings_ADDiscreteMC1_t.py $current_dir
fi

if [ "$MontecarloADtjac" = "true" ]; then
    echo "RUNNING MONTECARLOADTJAC"
    # python settings_ADDiscreteLoadsMC_validation.py $current_dir
    python settings_ADDiscreteMC1_t_fdjac.py $current_dir
fi

if [ "$MontecarloADtfd" = "true" ]; then
    echo "RUNNING MONTECARLOADTFD"
    # python settings_ADDiscreteLoadsMC_validation.py $current_dir
    python settings_ADDiscreteMC1_t_fd.py $current_dir
fi

if [ "$MontecarloADfem" = "true" ]; then
    echo "RUNNING MONTECARLOADFEM"
    python settings_ADDiscreteMC1_fem.py $current_dir
    #python settings_ADDiscreteLoadsMC.py $current_dir
fi

if [ "$Gust" = "true" ]; then
    echo "RUNNING GUST"
    python settings_gust1shard.py $current_dir
fi

if [ "$Forager" = "true" ]; then
    echo "RUNNING FORAGER"
    python settings_gustforager.py $current_dir
fi
