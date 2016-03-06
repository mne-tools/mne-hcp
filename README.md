# MNE-HCP

Python tools for processing HCP data using MNE-Python

## disclaimer and goals

- this code is under active, research-driven development and the API is still unstable.
- we only intend to support a subset of the files shipped with HCP. Precisely, for now it is not planned to support io and processing for any outputs of the HCP inverse pipelines.
- this library breaks with some MNE conventions due to peculiarities of the HCP data shipping policy. At a later stage this code will likely be wrapped by MNE-Python to provide a more common API.

## contributions
- as in MNE

## dependencies
- MNE master branch
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