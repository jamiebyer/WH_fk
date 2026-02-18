# FK Beamforming

**Description**
- fk beamforming
- Rayleigh wave dispersion curves

**Data**
- 2 WH stations
- figures of original results
- original max files
- original miniseeds
- txt files: 
    - coordinates
    - corrected coordinates
    - geopsy dispersion curve (from max2curve or gphistogram)

Can use dispersion curve from txt_file, or can compute dispersion curve from max file.


**Processing steps**
1. create database from miniseeds and coordinates ".gpy"
2. run fk-beamforming
`./geopsypack-src-3.5.2/bin/geopsy-fk`
`geopsy-fk -db Mirandola.gpy -group C_135_405-Z -param limits.param`