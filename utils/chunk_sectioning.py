"""
Utility to associate chunks with document sections and headers.
Enriches chunks with header hierarchy metadata and context.
"""

from typing import List, Tuple, Dict, Optional
import re


class ChunkSectionMapper:
    """
    Maps chunks to document headers and enriches chunks 
    with header hierarchy information.
    """

    def __init__(
        self,
        add_section_prefix: bool = True
    ):
        """
        Args:
            add_section_prefix: If True, prepend header hierarchy to chunk.
        """
        self.add_section_prefix = add_section_prefix
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def extract_headers_hierarchy(
        self,
        text: str
    ) -> List[Dict]:
        """
        Extract markdown headers with hierarchy information.
        
        Args:
            text: Full text content with markdown headers
            
        Returns:
            List of dicts with 'level', 'title', 'start_pos', 'end_pos'
        """
        headers = []
        
        for match in self.header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()
            
            headers.append({
                'level': level,
                'title': title,
                'start_pos': start_pos,
                'end_pos': None,
                'match_obj': match
            })
        
        # Set end positions based on where next header of same or higher level starts
        for i in range(len(headers)):
            # Find the next header at same or higher level
            next_end = len(text)
            
            current_level = headers[i]['level']
            
            for j in range(i + 1, len(headers)):
                if headers[j]['level'] <= current_level:
                    next_end = headers[j]['start_pos']
                    break
            
            headers[i]['end_pos'] = next_end
        
        return headers

    def get_header_hierarchy_for_position(
        self,
        position: int,
        headers: List[Dict],
        text_length: int
    ) -> List[str]:
        """
        Get the header hierarchy for a given text position.
        Returns complete parent-child hierarchy.
        
        Args:
            position: Character position in text
            headers: List of headers with positions
            text_length: Length of the text (for bounds checking)
            
        Returns:
            List of headers from top to bottom that contain this position
        """
        # Find all headers that contain this position
        containing_headers = []
        
        for header in headers:
            end_pos = header['end_pos'] if header['end_pos'] else text_length
            
            if header['start_pos'] <= position < end_pos:
                containing_headers.append(header)
        
        # Sort by level to get proper hierarchy (top-level first)
        containing_headers.sort(key=lambda x: x['level'])
        
        return [h['title'] for h in containing_headers]

    def map_chunks_to_headers(
        self,
        chunks: List[str],
        text: str,
        headers: List[Dict]
    ) -> Tuple[List[str], List[Dict]]:
        """
        Associate each chunk with its header hierarchy.
        
        Args:
            chunks: List of text chunks
            text: Full source text
            headers: List of headers extracted from text
            
        Returns:
            Tuple of (enhanced_chunks, metadata_list)
        """
        chunk_metadata = []
        enhanced_chunks = []
        
        text_length = len(text)
        
        # Build full text with position markers
        current_pos = 0
        chunk_positions = []
        
        for chunk in chunks:
            # Find where this chunk starts in the original text
            # Account for modifications (section prefixes, etc)
            chunk_clean = chunk.replace('**Headers: ', '').split('**\n\n', 1)
            if len(chunk_clean) > 1:
                chunk_clean = chunk_clean[1]
            else:
                chunk_clean = chunk
            
            # Find this chunk in original text
            search_start = current_pos
            pos = text.find(chunk_clean, search_start)
            
            if pos == -1:
                # Fallback: use first 100 chars to find approximate position
                search_text = chunk_clean[:100]
                pos = text.find(search_text, search_start)
            
            if pos == -1:
                pos = current_pos
            
            chunk_positions.append(pos)
            current_pos = pos + len(chunk_clean)
        
        # Map chunks to headers
        for chunk, chunk_pos in zip(chunks, chunk_positions):
            hierarchy = self.get_header_hierarchy_for_position(
                chunk_pos, 
                headers,
                text_length
            )
            
            header_str = " > ".join(hierarchy) if hierarchy else "Document"
            
            chunk_metadata.append({
                "section": header_str,
                "headers": hierarchy
            })
            
            if self.add_section_prefix and hierarchy:
                header_prefix = " > ".join(hierarchy)
                enhanced = f"**Headers: {header_prefix}**\n\n{chunk}"
                enhanced_chunks.append(enhanced)
            else:
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks, chunk_metadata

    def enrich_chunks(
        self,
        chunks: List[str],
        text: str,
        section_info: str = "Document"
    ) -> Tuple[List[str], List[Dict]]:
        """
        All-in-one method: extract headers and map chunks to headers.
        
        Args:
            chunks: List of text chunks
            text: Full source text  
            section_info: Ignored (kept for compatibility)
            
        Returns:
            Tuple of (enhanced_chunks, metadata_list)
        """
        headers = self.extract_headers_hierarchy(text)
        
        if not headers:
            # No headers found - return chunks as-is
            return chunks, [
                {"section": "Document", "headers": []} 
                for _ in chunks
            ]
        
        return self.map_chunks_to_headers(chunks, text, headers)
