# filenames = ['/home/devil/Thesis/brown/cf01', '/home/devil/Thesis/brown/cf02', '/home/devil/Thesis/brown/cf03',
#              '/home/devil/Thesis/brown/cf04', '/home/devil/Thesis/brown/cf05', '/home/devil/Thesis/brown/cf06',
#              '/home/devil/Thesis/brown/cf07', '/home/devil/Thesis/brown/cf08', '/home/devil/Thesis/brown/cf09',
#              '/home/devil/Thesis/brown/cf10', '/home/devil/Thesis/brown/cf11', '/home/devil/Thesis/brown/cf12',
#              '/home/devil/Thesis/brown/cf13', '/home/devil/Thesis/brown/cf14', '/home/devil/Thesis/brown/cf15',
#              '/home/devil/Thesis/brown/cf16', '/home/devil/Thesis/brown/cf17', '/home/devil/Thesis/brown/cf18',
#              '/home/devil/Thesis/brown/cf19', '/home/devil/Thesis/brown/cf20', '/home/devil/Thesis/brown/cf21',
#              '/home/devil/Thesis/brown/cf22', '/home/devil/Thesis/brown/cf23', '/home/devil/Thesis/brown/cf24',
#              '/home/devil/Thesis/brown/cf25', '/home/devil/Thesis/brown/cf26', '/home/devil/Thesis/brown/cf27']
# filenames = ['/home/devil/Thesis/brown/cf01', '/home/devil/Thesis/brown/cf02', '/home/devil/Thesis/brown/cf03',
#              '/home/devil/Thesis/brown/cf04', '/home/devil/Thesis/brown/cf05', '/home/devil/Thesis/brown/cf06',
#              '/home/devil/Thesis/brown/cf07', '/home/devil/Thesis/brown/cf08', '/home/devil/Thesis/brown/cf09',
#              '/home/devil/Thesis/brown/cf10', '/home/devil/Thesis/brown/cf11', '/home/devil/Thesis/brown/cf12',
#              '/home/devil/Thesis/brown/cf13', '/home/devil/Thesis/brown/cf14', '/home/devil/Thesis/brown/cf15',
#              '/home/devil/Thesis/brown/cf16', '/home/devil/Thesis/brown/cf17']
filenames = ['/home/devil/Thesis/brown/cf01', '/home/devil/Thesis/brown/cf02', '/home/devil/Thesis/brown/cf03',
             '/home/devil/Thesis/brown/cf04', '/home/devil/Thesis/brown/cf05', '/home/devil/Thesis/brown/cf06',
             '/home/devil/Thesis/brown/cf07', '/home/devil/Thesis/brown/cf08', '/home/devil/Thesis/brown/cf09',
             '/home/devil/Thesis/brown/cf10', '/home/devil/Thesis/brown/cf11', '/home/devil/Thesis/brown/cf12',
             '/home/devil/Thesis/brown/cf13', '/home/devil/Thesis/brown/cf14', '/home/devil/Thesis/brown/cf15',
             '/home/devil/Thesis/brown/cf16', '/home/devil/Thesis/brown/cf17', '/home/devil/Thesis/brown/cf18',
             '/home/devil/Thesis/brown/cf19', '/home/devil/Thesis/brown/cf20', '/home/devil/Thesis/brown/cf21',
             '/home/devil/Thesis/brown/cf22', '/home/devil/Thesis/brown/cf23', '/home/devil/Thesis/brown/cf24',
             '/home/devil/Thesis/brown/cf25', '/home/devil/Thesis/brown/cf26', '/home/devil/Thesis/brown/cf27',
             '/home/devil/Thesis/brown/cf28', '/home/devil/Thesis/brown/cf29', '/home/devil/Thesis/brown/cf30',
             '/home/devil/Thesis/brown/cf31', '/home/devil/Thesis/brown/cf32', '/home/devil/Thesis/brown/cf33',
             '/home/devil/Thesis/brown/cf34', '/home/devil/Thesis/brown/cf35', '/home/devil/Thesis/brown/cf36',
             '/home/devil/Thesis/brown/cf37', '/home/devil/Thesis/brown/cf38', '/home/devil/Thesis/brown/cf39',
             '/home/devil/Thesis/brown/cf40', '/home/devil/Thesis/brown/cf41', '/home/devil/Thesis/brown/cf42',
             '/home/devil/Thesis/brown/cf43', '/home/devil/Thesis/brown/cf44', '/home/devil/Thesis/brown/cf45',
             '/home/devil/Thesis/brown/cf46', '/home/devil/Thesis/brown/cf47', '/home/devil/Thesis/brown/cf48']

with open('/home/devil/Thesis/cf_all', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)