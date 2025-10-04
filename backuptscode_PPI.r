# Target Score package
library(targetscore)
library(dplyr)
source('./TargetScoreKorkut/my_calc_targetscore.r', chdir = TRUE)


args = commandArgs(trailingOnly = TRUE)

prot <- args[1] #'UACC257' #Type the cell line name or you can give cmd arguments from bash terminal with Rscript command args[1] 
name_map <- read.table("./sign-resp-data.txt", sep=",", col.names=c("name1", "name2"), stringsAsFactors=FALSE)
name_map <- setNames(name_map$name2, name_map$name1)

sign <- name_map[prot] #'SKCM' #Type the cell line name or you can give cmd arguments from bash terminal with Rscript command args[2]

print(paste(prot,' is selected.', sep=''))
#print(paste(sign,' is being used.', sep=''))

ab_file_path = "./ts-inputs/antibody_map_08272020.csv"
response_path = "./mean-filled-resps/"
f_score_path = "./ts-inputs/fs.csv"
sign_response_path = paste("./ts-inputs/signaling-responses/TCGA-",paste(sign,'-L4.csv',sep=''), sep='')
save_path = './ppi1perm/'

# Max distance
max_dist <- 1 # changing this value requires additional work to compute product(wk). This is not a priority
# verbose <- FALSE

mab_to_genes <- read.csv(ab_file_path,stringsAsFactors = FALSE)

# Read drug perturbation data for BT474
#proteomic_responses2 <- read.csv(paste("./cek-research/actual-responses/",prot,'.csv', sep=''), header = TRUE, sep = ",")
proteomic_responses2 <- read.csv(paste(response_path,prot,'.csv', sep=''), header = TRUE, sep = ",")

uid_names = data.frame("UID" = proteomic_responses2[, 2])

proteomic_responses <- proteomic_responses2[ , -1]

#proteomic_responses <- proteomic_responses2[ , -c(2, 3, 4, 5, 6)]
#print(proteomic_responses)
#asddasdfafs

fs_override <- read.csv("./TargetScoreKorkut/inst/test_data/fs_mod.csv", header = TRUE, stringsAsFactors = FALSE)

sanitize <- function(x) {
  # Replace spaces, underscores, dots, and dashes and convert to lowercase
  gsub("[ ._-]", "", tolower(x))
}

# Assuming df1 and df2 are your dataframes, and array2 is a column in df2
array1 <- colnames(proteomic_responses)
array2 <- mab_to_genes$'AntibodyLabel'  # replace 'column_name' with the actual column name in df2

# Sanitize both arrays
sanitized_array1 <- sapply(array1, sanitize)
sanitized_array2 <- sapply(array2, sanitize)

# Create a named vector for mapping: names from sanitized_array2, values from array2

mapping <- setNames(array2, sanitized_array2)
# Find sanitized matches
matches <- sanitized_array1 %in% sanitized_array2
# Map the matching column names in df1 to their counterparts in array2
colnames(proteomic_responses)[matches] <- mapping[sanitized_array1[matches]]

# df1 now has its column names converted to their counterparts in array2 where sanitized versions match

idx_ab_map <- which(mab_to_genes[, 1] %in% colnames(proteomic_responses))

diffs <- setdiff(colnames(proteomic_responses), unique(mab_to_genes[idx_ab_map, 1]))

proteomic_responses <- proteomic_responses[, !colnames(proteomic_responses) %in% diffs]

idx_ab_map2 <- which(colnames(proteomic_responses) %in% mab_to_genes[, 1])

diffs2 <- setdiff(colnames(proteomic_responses), unique(mab_to_genes[idx_ab_map, 1]))

proteomic_responses <- proteomic_responses[, !colnames(proteomic_responses) %in% diffs2]

n_prot <- dim(proteomic_responses)[2]

array1 <- colnames(proteomic_responses)
array2 <- fs_override$prot  # replace 'column_name' with the actual column name in df2
# Sanitize both arrays
sanitized_array1 <- sapply(array1, sanitize)
sanitized_array2 <- sapply(array2, sanitize)

# Create a named vector for mapping: names from sanitized_array2, values from array2
mapping <- setNames(array2, sanitized_array2)

# Find sanitized matches
matches <- sanitized_array1 %in% sanitized_array2

# Map the matching column names in df1 to their counterparts in array2
colnames(proteomic_responses)[matches] <- mapping[sanitized_array1[matches]]

idx_ab_map <- which(mab_to_genes[, 1] %in% colnames(proteomic_responses))
diffs <- setdiff(colnames(proteomic_responses), unique(mab_to_genes[idx_ab_map,1]))

proteomic_responses <- proteomic_responses[, !colnames(proteomic_responses) %in% diffs]
n_prot <- dim(proteomic_responses)[2]
#print(diffs)

#print(unique(mab_to_genes[idx_ab_map,1]))
#print(colnames(proteomic_responses))
print(length(unique(mab_to_genes[idx_ab_map,1])))
print(n_prot)
#print(colnames(proteomic_responses))

#sanitized_array11 <- sapply(colnames(proteomic_responses), sanitize)
#sanitized_array22 <- sapply(unique(mab_to_genes[idx_ab_map,1]), sanitize)

#print(sanitized_array11)
#print(matches)
print('TEST 1 completed')

print('TEST 2 completed')
#saveRDS(network, file.path(output_dir, "bt474_bionetwork.rds"))

# Read Global Signaling file for BRCA
signaling_responses <- read.csv(sign_response_path, row.names = 1)

signaling_responses <- signaling_responses[, 4:ncol(signaling_responses)]
signaling_responses[is.na(signaling_responses)] <- 0
print('TEST 3 completed')
# Extract network
#network <- targetscore::predict_dat_network(
#  data <- signaling_responses,
#  n_prot = n_prot,
#  proteomic_responses = proteomic_responses
#)
print('TEST 4 completed')
print(nprot)
print(dim(proteomic_responses))

#saveRDS(network, file.path(output_dir, "bt474_datnetwork.rds"))
# Extract protein-protein interaction knowledge
network <- targetscore::predict_bio_network(
  n_prot = n_prot,
  proteomic_responses = proteomic_responses,
  max_dist = max_dist,
  mab_to_genes = mab_to_genes
)
#print('TEST 5 completed')
#prior <- network$wk
#print('TEST 6 completed')
# pr-glasso
#network <- targetscore::predict_hybrid_network(
#  data = signaling_responses,
#  prior = prior,
#  n_prot = n_prot,
#  proteomic_responses = proteomic_responses,
#  mab_to_genes = mab_to_genes
#)
print('TEST 7 completed')
#saveRDS(network, file.path(output_dir, "bt474_hybnetwork.rds"))
#print(matching_rows)
# Extract functional score
fs <- targetscore::get_fs_vals(
  n_prot = n_prot,
  proteomic_responses = proteomic_responses,
  mab_to_genes = mab_to_genes,
  fs_override = fs_override
)

print('TEST 8 completed')
#saveRDS(fs, file.path("./cek-research/ts-outputs/fs-rds", paste(prot,"_fs.rds", sep='')))
print('TEST 9 completed')
## Calculate Target Score

# Permutation times
n_perm <- 1

# Reference network
wk <- network$wk
wks <- network$wks
dist_ind <- network$dist_ind
inter <- network$inter
edgelist <- network$edgelist
print('TEST 10 completed')
# Number of Conditions needed to calculate
n_cond <- nrow(proteomic_responses)
print('TEST 11 completed')
# Set initial value
ts <- matrix(NA,
             nrow = n_cond, 
             ncol = n_prot,
             dimnames = list(
               rownames(proteomic_responses),
               colnames(proteomic_responses)
             )
)
print('TEST 12 completed')
ts_p <- matrix(NA,
               nrow = n_cond, 
               ncol = n_prot,
               dimnames = list(
                 rownames(proteomic_responses),
                 colnames(proteomic_responses)
               )
)
print('TEST 13 completed')
ts_q <- matrix(NA,
               nrow = n_cond, 
               ncol = n_prot,
               dimnames = list(
                 rownames(proteomic_responses),
                 colnames(proteomic_responses)
               )
)
print('TEST 14 completed')
# Calculate for each row entry separately
for (i in 1:n_cond) {
  results <- my_calc_targetscore(
    wk = wk,
    wks = wks,
    dist_ind = dist_ind,
    edgelist = edgelist,
    n_dose = 1,
    n_prot = n_prot,
    proteomic_responses = proteomic_responses[i, ],
    fs_dat = fs,
    verbose=FALSE
  )
  
  ts[i, ] <- as.numeric(results$ts) # as.numeric needed if multiple conditions
}
print('TEST 15 completed')

targetscore_output = ts

print(dim(targetscore_output))
