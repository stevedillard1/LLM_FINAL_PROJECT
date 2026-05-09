from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
d = get_pretrained_saes_directory()
release = d['gemma-scope-2-4b-it-res']
hooks = list(release.saes_map.keys())
print(f'Total hooks: {len(hooks)}')
print('Sample hooks:', hooks[:5])
