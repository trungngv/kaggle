library(rmarkdown)
source('src/common.R')
render('src/explore.Rmd', output_format = "html_document", output_file = 'renthop.html')
