#!/usr/bin/python3

"Download ECMA specification from https://tc39.es/ecma262/multipage/ and convert it to XML, as it is some mixed nonstandard HTML/SGML."

import requests
from pathlib import Path
from lxml.etree import fromstring, tounicode


def download_page(link, spec_dir):
	r = requests.get(link)
	
	text = r.text
	del r
	
	links = set()
	m = 0
	while (m := text.find('<a href=', m)) != -1:
		assert text[m + 8] == '"'
		m += 9
		n = text.index('"', m)
		href = text[m:n]
		if href.startswith('#') or href.startswith('.'):
			pass
		else:
			href = href.split('#')[0]
			if href.endswith('.html') and not (href.startswith('http://') or href.startswith('https://')):
				links.add(href)
		m = n + 1

	m = text.rindex('</emu-')
	n = text.index('>', m) + 1
	end_tag = text[m:n]
	start_tag = end_tag.replace('/', '').replace('>', '')
	
	m = text.index(start_tag)
	text = text[m:n]
	
	text = text.replace('<emu-', '<emu:')
	text = text.replace('</emu-', '</emu:')
	text = text.replace('<br>', '<br/>')
	text = text.replace('&nbsp;', '&#160;')
	
	m = 0
	while (m := text.find('<img ', m)) != -1:
		n = text.index('>', m)
		text = text[:n] + '/' + text[n:]
		m += 1
	
	m = text.index('>')
	text = text[:m] + ' xmlns:emu="https://github.com/rbuckton/grammarkdown" xmlns="http://www.w3.org/1999/xhtml"' + text[m:]
	
	tree = fromstring(text)
	name = tree.attrib['id']
	if name.startswith('sec-'):
		name = name[4:]
	
	try:
		secnum = tree.xpath('.//*[@class="secnum"]')[0].text
		secnum = int(secnum)
	except IndexError:
		name = "00-" + name
	except ValueError:
		name = secnum + "-" + name
	else:
		name = f"{secnum:02d}-" + name
	
	text = '<?xml version="1.0"?>\n' + text
	(spec_dir / (name + '.xml')).write_text(text)
	
	return links


if __name__ == '__main__':
	baseurl = 'https://tc39.es/ecma262/multipage/'
	spec_dir = Path('specification')
	spec_dir.mkdir(exist_ok=True)
	possible_links = set({''})
	visited_links = set()
	
	while unvisited_links := possible_links - visited_links:
		for link in unvisited_links:
			print(baseurl + link)
			links = download_page(baseurl + link, spec_dir)
			possible_links.update(links)
			visited_links.add(link)

