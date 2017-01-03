# init
libs <- c("tm", "plyr", "class")
lapply(libs, require, character.only = TRUE)

# set options
options(stringsAsFactors = FALSE)

# set parameters
pathname <- "C:/Users/zallen/Documents/GitHub/pci-email-classification/emails"
pathname.test <- sprintf("%s/%s", pathname, "test")
pathname.train <- sprintf("%s/%s", pathname, "train")
targets <- list.dirs(path = pathname.train, full.names = FALSE, recursive = FALSE)
testCount <- length(list.files(path = pathname.test, recursive = TRUE))

# clean text
removeHtmlTags <- content_transformer(function(x, pattern) {
  return (gsub("<.*?>", "", x))
})

cleanCorpus <- function(corpus) {
  corpus.tmp <- corpus
  corpus.tmp <- tm_map(corpus.tmp, removeHtmlTags, mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, removePunctuation, mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace, mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower), mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"), mc.cores=1)
  return(corpus.tmp)
}


# build TDM
generateTDM <- function(target, path) {
  s.dir <- sprintf("%s/%s", path, target)
  s.cor <- Corpus(DirSource(directory = s.dir, encoding = "UTF-8"))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  
  s.tdm <- removeSparseTerms(s.tdm, 0.7)
  return <- list(name = target, tdm = s.tdm)
}

tdm <- lapply(targets, generateTDM, path = pathname.train)
tdm <- append(tdm, lapply(targets, generateTDM, path = pathname.test))

# attach name
bindTargetToTDM <- function(tdm) {
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringsAsFactors = FALSE)
  
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "__target"
  return(s.df)
}

targetTDM <- lapply(tdm, bindTargetToTDM)

# stack
tdm.stack <- do.call(rbind.fill, targetTDM)
tdm.stack[is.na(tdm.stack)] <- 0

# hold-out
train.idx <- (1:(nrow(tdm.stack)-testCount))
test.idx <- ((nrow(tdm.stack)-testCount + 1):nrow(tdm.stack))

# model - KNN
tdm.target <- tdm.stack[, "__target"]
tdm.stack.nl <- tdm.stack[, !colnames(tdm.stack) %in% "__target"]

knn.pred <- knn(tdm.stack.nl[train.idx, ], tdm.stack.nl[test.idx, ], tdm.target[train.idx])

# results
testTdm.idx <- (((length(tdm)/2)+1):(length(tdm)))
testTdm <- tdm[testTdm.idx]
correct <- 0
targetIdx <- 1

testDocs <- c()
actualTargets <- c()
for (idx in (1:length(targets))) {
  for (d in (1:length(testTdm[[idx]]$tdm$dimnames$Docs))) {
      testDocs <- c(testTdm[[idx]]$tdm$dimnames$Docs[d], as.list(testDocs))
      actualTargets <- c(testTdm[[idx]]$name, as.list(actualTargets))
  }
}

for (idx in (1:length(knn.pred))) {
  r <- testDocs[idx]
  r <- paste(r, "pred:", knn.pred[idx])
  r <- paste(r, "; actual:", actualTargets[idx], ";")
  if (knn.pred[idx] == actualTargets[idx]) correct <- correct + 1
  print(r)
}

print(paste(correct, "out of", length(knn.pred), "correct", "-", (correct/length(knn.pred)*100), "%"))
