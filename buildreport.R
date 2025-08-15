# Load packages
require(knitr)
require(markdown)

# Create .md, .html, and .pdf files for report 1
knit("report.Rmd")
markdownToHTML('report.md', 'report.html', options=c("use_xhml"))
system("pandoc -s report.html -o report.pdf")