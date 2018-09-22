library(tidyverse)
dt = read.csv('claims_final.csv')
fm_id_group <- dt%>%group_by(family_id)%>%summarise(family_total_claim=n())
dt_f <- inner_join(dt, fm_id_group)
fm_gpm_group <- dt%>%group_by(family_id, member_id)%>%summarise(individual_frequency = n())
dt_f <- inner_join(dt_f, fm_gpm_group)


provider_id_group <- dt %>% group_by(provider_id)%>%summarise(provider_num = n())

provider_id_type_group <- dt %>% group_by(provider_id, provider_type) %>% summarise(type_num = n())
provider_id_type_joined <- inner_join(provider_id_type_group, provider_id_group)
provider_id_type_joined <- provider_id_type_joined %>% mutate(provider_type_percentage = provider_num / type_num)
dt_f <-inner_join(dt_f, provider_id_type_joined%>%select(provider_id, provider_type, provider_type_percentage))

provider_state_grouped <- dt %>% group_by(provider_id, state_code)%>%summarise(state_num = n())
provider_state_id_joined <- inner_join(provider_id_group, provider_state_grouped)%>%mutate(provider_state_percentage = state_num / provider_num)
provider_state_id_joined%>%filter(provider_state_percentage!=1)

type_grouped <- dt %>% group_by(provider_type)%>% summarise(type_num = n())
type_procedure_grouped <- dt %>% group_by(provider_type, medical_procedure_code) %>% summarise(procedure_num = n())
type_procedure_joined <- inner_join(type_procedure_grouped, type_grouped) %>% mutate(procedure_type_percentage = procedure_num / type_num)
dt_f <- inner_join(dt_f, type_procedure_joined%>%select(provider_type, medical_procedure_code, procedure_type_percentage))

procedure_avg_grouped <- dt %>% group_by(medical_procedure_code) %>% summarise(avg_amount_procedure = mean(amount))
dt_f <- inner_join(dt_f, procedure_avg_grouped)


procedure_provider_grouped <- dt %>% group_by(medical_procedure_code, provider_id)%>% summarise(procedure_num = n())
procedure_provider_joined <- inner_join(procedure_provider_grouped, provider_id_group) %>% mutate(procedure_provider_percentage = procedure_num / provider_num)
dt_f <- inner_join(dt_f, procedure_provider_joined%>%select(medical_procedure_code, provider_id, procedure_provider_percentage))

dt_f <- inner_join(dt_f,dt_f%>%group_by(provider_type)%>%summarise(mean_procedure_type_percentage = mean(procedure_type_percentage)))
dt_f <- inner_join(dt_f, dt_f%>%group_by(provider_id)%>%summarise(mean_procedure_provider_percentage = mean(procedure_provider_percentage)))

dt_f <- dt_f %>% select(-provider_type_percentage)

dt_f <- dt_f %>% mutate(normalized_procedure_type_percentage=procedure_type_percentage / mean_procedure_type_percentage)
dt_f <- dt_f %>% mutate(normalized_procedure_provider_percentage=procedure_provider_percentage / mean_procedure_provider_percentage)
dt_f <- dt_f %>% mutate(normalized_money_difference = amount-avg_amount_procedure)

dt_f <- dt_f %>% mutate(month = floor((date_of_service %% 10000)/ 100))

month_grouped <- dt_f %>% group_by(month) %>% summarise(month_cnt = n())
month_procedure_grouped <- dt_f %>% group_by(month, medical_procedure_code) %>% summarise(month_procedure_cnt = n())
month_procedure_joined <- inner_join(month_procedure_grouped, month_grouped)%>%mutate(month_procedure_percentage = month_procedure_cnt/month_cnt)
dt_f<-inner_join(dt_f, month_procedure_joined%>%select(month, medical_procedure_code, month_procedure_percentage))

dt_f <- inner_join(dt_f, dt_f%>%group_by(month)%>%summarise(mean_month_procedure_percentage = mean(month_procedure_percentage)))
dt_f <- dt_f%>% mutate(normalized_month_procedure_percentage = month_procedure_percentage / mean_month_procedure_percentage)



dt_kmeans <- kmeans(dt_f%>%select(family_total_claim, individual_frequency, normalized_procedure_type_percentage, normalized_procedure_provider_percentage, normalized_money_difference, normalized_month_procedure_percentage), 2, nstart = 5, iter.max=10)
dt_kmeans_result <- dt_f
dt_kmeans_result$cluster <- dt_kmeans$cluster

python_model_result <- read.csv('python_scores.csv')
dt_kmeans_result$score <- python_model_result$score
dt_kmeans_result%>%ggplot(aes(x=score)) + geom_histogram() + facet_grid(~cluster)

dt_f_normed <- dt_f%>%select(family_total_claim, individual_frequency, normalized_procedure_type_percentage, normalized_procedure_provider_percentage, normalized_money_difference, normalized_month_procedure_percentage)%>%mutate(family_total_claim = (family_total_claim-mean(family_total_claim))/sd(family_total_claim))%>%mutate(individual_frequency=(individual_frequency-mean(individual_frequency))/sd(individual_frequency))%>%mutate(normalized_procedure_type_percentage=(normalized_procedure_type_percentage-mean(normalized_procedure_type_percentage))/sd(normalized_procedure_type_percentage))
dt_f_normed <- dt_f_normed %>% mutate(normalized_procedure_provider_percentage = (normalized_procedure_provider_percentage-mean(normalized_procedure_provider_percentage))/sd(normalized_procedure_provider_percentage))%>%mutate(normalized_money_difference=(normalized_money_difference-mean(normalized_money_difference))/sd(normalized_money_difference))%>%mutate(normalized_month_procedure_percentage=(normalized_month_procedure_percentage-mean(normalized_month_procedure_percentage))/sd(normalized_month_procedure_percentage))
dt_kmeans_normed<-kmeans(dt_f_normed, 2, nstart = 5, iter.max=10)
dt_kmeans_normed_result<-dt_f
dt_kmeans_normed_result$score<-python_model_result$score
dt_kmeans_normed_result$cluster<-dt_kmeans_normed$cluster
dt_kmeans_normed_result%>%ggplot(aes(x=score)) + geom_histogram() + facet_grid(~cluster)

dt_predicted<-dt
dt_predicted$score <- python_model_result$score
dt_f1<-dt_predicted%>%group_by(provider_id)%>%summarise(max_score=max(score))%>%arrange(desc(max_score))


dt_f2<-dt
dt_f2$score <- python_model_result$score
dt_f2<-dt_f2%>%group_by(family_id, member_id, provider_id, provider_type, date_of_service)%>%summarise(score=max(score))
dt_f2<- dt_f2%>%arrange(desc(score))
dt_f2$rank <- c(1: nrow(dt_f2))
