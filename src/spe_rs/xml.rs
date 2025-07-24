use std::{collections::HashMap, error::Error};

#[derive(Debug)]
pub struct XMLTag {
    pub name: String,
    pub parameters: HashMap<String, String>,
    pub contents: String,
    pub children: Vec<XMLTag>,
}

impl<'a> XMLTag {
    pub fn from_str(xml_header: &str) -> Result<XMLTag, Box<dyn Error>> {
        let mut chars = xml_header.chars().peekable();

        // Fast forward to first opening tag
        while let Some(ch) = chars.peek() {
            match ch {
                '<' => break,
                _ => {
                    chars.next();
                }
            }
        }

        let mut stack: Vec<XMLTag> = Vec::new();

        while let Some(ch) = chars.next() {
            let Some(next_ch) = chars.peek() else {
                return Err("XML data depleted before root was closed".into());
            };
            // We figure out the structure of the XML document by looking at
            // the next two characters ...
            match (ch, next_ch) {
                // Opening Tag
                ('<', 'a'..='z') | ('<', 'A'..='Z') => {
                    let tagname: String = chars.by_ref().take_while(|&ch| ch != '>').collect();
                    let (name, parameters) = parse_tagname(&tagname);
                    if !tagname.ends_with('/') {
                        stack.push(XMLTag {
                            name,
                            parameters,
                            contents: String::new(),
                            children: Vec::new(),
                        });
                    } else {
                        // If this is a single tag (without closing tag) we add it to the current children
                        let tag = XMLTag {
                            name,
                            parameters,
                            contents: "".to_string(),
                            children: Vec::with_capacity(0),
                        };
                        if let Some(parent_tag) = stack.last_mut() {
                            parent_tag.children.push(tag);
                        }
                    }
                }
                // Closing Tag
                ('<', '/') => {
                    let _: String = chars.by_ref().take_while(|&ch| ch != '>').collect();

                    let current_tag = stack.pop();

                    if let Some(tag) = current_tag {
                        if let Some(parent_tag) = stack.last_mut() {
                            parent_tag.children.push(tag);
                        } else {
                            // If there are no more children, we reached the root tag
                            return Ok(tag);
                        }
                    } else {
                        return Err("Stack depleted before XML root reached".into());
                    };
                }
                // Raw contents between two tags
                (_, _) => {
                    let mut current_contents = String::new();
                    current_contents.push(ch);
                    while let Some(ch_peeked) = chars.peek() {
                        match ch_peeked {
                            // The next tag beginns, stop consuming characters
                            '<' => break,
                            _ => {
                                // All other characters are consumed as raw inner tag contents
                                current_contents.push(*ch_peeked);
                                chars.next();
                            }
                        }
                    }
                    if let Some(parent_tag) = stack.last_mut() {
                        parent_tag.contents += &current_contents;
                    }
                }
            }
        }
        Err("XML data depleted before root was closed".into())
    }

    /// Build a hashmap to conveiniently access data from XML footer
    pub fn build_index(&'a self) -> HashMap<String, &'a XMLTag> {
        let mut index = HashMap::new();
        let mut stack: Vec<_> = self
            .children
            .iter()
            .map(|ch| (ch, self.name.clone()))
            .collect();
        while let Some((tag_ref, base_name)) = stack.pop() {
            let key = base_name + "/" + &tag_ref.name;
            for ch in tag_ref.children.iter() {
                stack.push((ch, key.clone()));
            }
            if let Some(_entry) = index.insert(key.clone(), tag_ref) {
                // TODO: Can we do something about non-unique keys?
                // eprintln!("Warning: overwriting key {key}");
            };
        }
        index
    }
}

fn parse_tagname(raw_contents: &str) -> (String, HashMap<String, String>) {
    let mut parts = raw_contents.split(" ");
    let Some(name) = parts.next() else {
        panic!("XML tag contained no valid name")
    };

    let mut params = HashMap::new();
    for raw_param in parts {
        if raw_param == "/" {
            // Single `/` indicates single tag, so this is a valid character we can ignore
            break;
        }
        if let Some((key, value)) = raw_param.split_once("=") {
            params.insert(key.to_owned(), trim_quotes(value).to_owned());
        } else {
            params.insert(raw_param.to_string(), "".to_string());
        }
    }
    (name.to_string(), params)
}

fn trim_quotes(s: &str) -> &str {
    s.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(s)
}

fn _debug_print_xml(tag: &XMLTag, indentation: usize) {
    let width = 2;
    let indent = std::iter::repeat_n(' ', indentation * width).collect::<String>();
    let extra_indent = std::iter::repeat_n(' ', width).collect::<String>();

    // Tagname
    println!("{indent}{}", tag.name);
    for (k, v) in tag.parameters.iter() {
        if !v.is_empty() {
            println!("{indent}{extra_indent}{k} = {v}")
        } else {
            println!("{indent}{extra_indent}{k}")
        }
    }
    if !tag.contents.is_empty() {
        println!("{indent}{extra_indent}contents = {}", tag.contents)
    }
    for child in tag.children.iter() {
        _debug_print_xml(child, indentation + 1);
    }
}
