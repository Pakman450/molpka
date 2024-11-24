

## To create environment with .yml file

```
conda env create -f environment.yml
```

## To update environment after adding new packages for conda

```
conda activate myenv
conda env update --file local.yml --prune
```