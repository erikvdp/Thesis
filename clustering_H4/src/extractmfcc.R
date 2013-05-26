library(tuneR)
infiles <- dir(path="/Users/Enrico/Google Drive/Thesis/clusterdata/extracted_clips/",pattern='\\.wav$') #get a list of all wav files
for(file in infiles) {
  digitalid = sub("[:puntct:].*$", "", file)
  mpfile <-file.path(getwd(), file)
  Waveobj = readWave(mpfile)
  mfcc = melfcc(Waveobj,wintime = 0.025, hoptime = 0.01,frames_in_rows=FALSE) 
  der = deltas(mfcc)
  features = t(rbind(mfcc,der))
  outputfile <- paste0(digitalid,"txt")
  if(nrow(features) == 2905)
    print(outputfile)
  write.table(features,file=paste0("/Users/Enrico/Google Drive/Thesis/clusterdata/mfcctxt/",paste0(digitalid,"txt")), row.names = FALSE, col.names = FALSE)
}