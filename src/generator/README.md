# Dataset Generator
This folder holds the generator code written to create a dataset to test our implementation. 

## Files
1. `generate.cpp`: generates a dataset depending on the `opts` data structure.

## Description 
The `opts` data structure is as follows:
```cpp
    struct opts {
        int n_files_start;
        int n_files_end;
        int step;
        int m_batches;
        string path;

        int minlength;
        int maxlength;
    };
```

1. `n_files_start`: Number of files to start with. The generator would generate `n_files_start` number of files in the first iteration.
2. `n_files_end`: Number of files to end with. The last iteration would generate `n_files_end` number of files.
3. `step`: The step jump to be taken at every iteration. If the first iteration generated `x` then the next generation would generate `x + step`files.
4. `m_batches`: The number of batches per step. This is done to calculate the _average_ speed per step. Every batch contains the current `n` number of files.
5. `path`: The output directory.
6. `minlength`: minimum size of generated file.
7. `maxlength`: maximum size of generated file.

## Instructions
1. Modify the `opts vars` object in `main` to your needs. 
2. `g++ generate.cpp`
3. `./a.out`

## Potential improvement
Make it accept nice parsed commandline arguments.