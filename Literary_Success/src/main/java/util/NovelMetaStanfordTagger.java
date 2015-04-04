package util;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class NovelMetaStanfordTagger {
	public static void main(String args[]) {
		MaxentTagger tagger = new MaxentTagger("taggers/bidirectional-distsim-wsj-0-18.tagger");
		//
		// The sample string
		String sample = "This is a sample text";

		// The tagged string
		String tagged = tagger.tagString(sample);

		// Output the result
		System.out.println(tagged);
	}
}
