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
testCount <- length(list.files(path = pathname.train, recursive = TRUE))

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
  
  s.tdm <- removeSparseTerms(s.tdm, 0.3)
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
# train.idx <- sample(nrow(train.tdm.stack), ceiling(nrow(train.tdm.stack) * 0.7))
# test.idx <- (1:nrow(tdm.stack)) [- train.idx]
train.idx <- (1:(nrow(tdm.stack)-testCount))
test.idx <- ((nrow(tdm.stack)-testCount + 1):nrow(tdm.stack))

# model - KNN
tdm.target <- tdm.stack[, "__target"]
tdm.stack.nl <- tdm.stack[, !colnames(tdm.stack) %in% "__target"]

knn.pred <- knn(tdm.stack.nl[train.idx, ], tdm.stack.nl[test.idx, ], tdm.target[train.idx])

# accuracy
conf.mat <- table("Predictions" = knn.pred, Actual = tdm.target[test.idx])

(accuracy <- sum(diag(conf.mat)) / length(test.idx) * 100)