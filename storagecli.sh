az storage account list -o Table --query "[].{Name:name, PrimaryLocation:primaryLocation, EndPoint:primaryEndpoints.blob}"

az storage account keys list -g sina -n godzillasinastorage -o Table --query "[].{Key:keyName, Value:value}"

# get the key
resource_group=sina
account_name=godzillasinastorage
container_name=dataset
directory_name=~/cloudfiles/data/cats_and_dogs_filtered/*


account_key=$(az storage account keys list -g $resource_group -n $account_name -o tsv --query "[0].{Value:value}")
az storage container create --name $container_name --auth-mode key --account-key $account_key --account-name $account_name -g $resource_group
az storage blob directory upload -c $container_name --auth-mode key --account-key $account_key --account-name $account_name -s $directory_name -d . --recursive