import csv
import os

INPUT = 'bbc-news-data.csv'
OUTDIR = 'data_berita'

def safe_filename(name):
    return name.replace('/', '_').replace('\\', '_')

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    count = 0
    created = []
    with open(INPUT, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row:
                continue
            # Expecting: category, filename, title, content...
            if len(row) < 2:
                continue
            filename = row[1].strip() if row[1].strip() else f'unnamed_{count:04d}.txt'
            filename = safe_filename(filename)
            title = row[2].strip() if len(row) > 2 else ''
            if len(row) > 3:
                # join any extra columns as the body
                body = '\n'.join(col for col in row[3:])
            else:
                body = ''
            if body:
                text = title + "\n\n" + body if title else body
            else:
                text = title

            outpath = os.path.join(OUTDIR, filename)
            with open(outpath, 'w', encoding='utf-8') as out:
                out.write(text)
            created.append(outpath)
            count += 1

    print(f'Wrote {count} files to "{OUTDIR}"')
    for p in created[:10]:
        print('  -', p)
    if len(created) > 10:
        print(f'  ...and {len(created)-10} more')

if __name__ == '__main__':
    main()
