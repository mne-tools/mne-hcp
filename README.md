# MNE-HCP

Python tools for processing HCP data using MNE-Python

## disclaimer and goals

This code is under active, research-driven development and the API is still unstable. At a later stage this code will likely be wrapped by MNE-Python to provide a more common API. For now consider the following caveats:
- we only intend to support a subset of the files shipped with HCP. Precisely, for now it is not planned to support io and processing for any outputs of the HCP inverse pipelines.
- the code is not covered by unit tests so far as I did not have the time to create mock testing data.
- this library breaks with some of MNE conventions due to peculiarities of the HCP data shipping policy. The basic IO is based on paths, not on files.
The core API is a file map, a list of dictionaries that map all available files
that we have for a given subject. This is needed because the information required to construct MNE objects is scattered across several files.

## contributions
- currently `@dengemann` is pushing frequently to master, if you plan to contribute, open issues and pull requests, or contact `@dengemann` directly. Discussions are welcomed.

## dependencies
- MNE-Python master branch
- scipy
- numpy
- matplotlib

## usage

The following data layout is expected. A folder that contains the HCP data
as they are unpacked by a zip, subject wise. See command that will produce this
layout.

```bash
for fname in $(ls *zip); do
    echo unpacking $fname;
    unzip -o $fname; rm $fname;
done
```

Note. This code still contains a second API that supports direct reading from
zipfiles. It will probably be removed as it turns out that zip files tend
to break and hence are not the best way to keep data.

# acknowledgements

This project is supported by the AWS Cloud Credits for Research program.
