# track-ml

Output of Data Preparation is a CSV file 

Each line is a particle and several hits of this particle 

Each line has 121 columns organized in the following way:

  - first 6 columns is particle information: tx, ty, tx, px, py, pz

  - 19 sets of 6 columns representing hits: tx,ty,tz, ch0, ch1, value
  
If the has less than 19 hits the remaings hits is fullfiled with zeros

the last colum 121 has 1 for fake hist and 0 for real hit

